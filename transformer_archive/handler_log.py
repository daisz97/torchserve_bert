from abc import ABC
import json
import logging
import os
import ast
import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
)
import numpy as np
import time
from ts.torch_handler.base_handler import BaseHandler
import re
import string

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithmfor Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        logger.info("ctx.manifest: %s", self.manifest)
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        logger.info("model_dir: %s", model_dir)
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        # Loading the shared object of compiled Faster Transformer Library if Faster Transformer is set
        if self.setup_config["FasterTransformer"]:
            faster_transformer_complied_path = os.path.join(model_dir, "libpyt_fastertransformer.so")
            torch.classes.load_library(faster_transformer_complied_path)

        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path, map_location=self.device)
        elif self.setup_config["save_mode"] == "pretrained":
            if self.setup_config["mode"] == "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir
                )
            elif self.setup_config["mode"] == "question_answering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
            elif self.setup_config["mode"] == "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            else:
                logger.warning("Missing the operation mode.")
            self.model.to(self.device)

        else:
            logger.warning("Missing the checkpoint or state_dict.")

        if any(fname for fname in os.listdir(model_dir) if fname.startswith("vocab.") and os.path.isfile(fname)):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, do_lower_case=self.setup_config["do_lower_case"]
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.setup_config["model_name"],
                do_lower_case=self.setup_config["do_lower_case"],
                use_fast=True
            )

        self.model.eval()

        logger.info(
            "Transformer model from path %s loaded successfully", model_dir
        )

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        # Question answering does not need the index_to_name.json file.
        if not self.setup_config["mode"] == "question_answering":
            if os.path.isfile(mapping_file_path):
                with open(mapping_file_path) as f:
                    self.mapping = json.load(f)
            else:
                logger.warning("Missing the index_to_name.json file.")
        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        input_ids_batch = None
        attention_mask_batch = None
        piece2word = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode('utf-8')
            # if self.setup_config["captum_explanation"] and not self.setup_config["mode"] == "question_answering":
            #     input_text_target = ast.literal_eval(input_text)
            #     input_text = input_text_target["text"]
            max_length = self.setup_config["max_length"]
            logger.info("Received text: '%s'", input_text)
            # preprocessing text for token_classification.
            # 1、filter string constant
            pattern = re.compile(r'\"(.*?)\"|\'(.*?)\'')  # 最短匹配：https://www.cnblogs.com/baxianhua/p/8571967.html
            input_text = re.sub(pattern, "", input_text)  # https://www.lidihuo.com/python/python-reg-expression.html
            print("input after string eliminated:", input_text)
            logger.info("input after string eliminated:'%s'", input_text)
            # 2、form input id
            inputs = self.tokenizer.encode_plus(input_text, max_length=int(max_length), pad_to_max_length=False,
                                                add_special_tokens=True, return_tensors='pt') #pad_to_max_length=True

            input_ids = inputs["input_ids"].to(self.device)
            piece2word = inputs.words()  # 值是从0开始，[CLS]和[SEP]是None
            attention_mask = inputs["attention_mask"].to(self.device)
            # making a batch out of the recieved requests
            # attention masks are passed for cases where input tokens are padded.
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat((attention_mask_batch, attention_mask), 0)
        return input_ids_batch, attention_mask_batch, piece2word

    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        input_ids_batch, attention_mask_batch, piece2word = input_batch
        inferences = []

        # Handling inference for token_classification.
        if self.setup_config["mode"] == "token_classification":
            outputs = self.model(input_ids_batch, attention_mask_batch)[0]
            print("This is the output size from the token classification model", outputs.size())
            print("This is the output from the token classification model", outputs)
            num_rows = outputs.shape[0]
            for i in range(num_rows):
                output = outputs[i].unsqueeze(0)
                predictions = torch.argmax(output, dim=2)
                tokens = self.tokenizer.tokenize(self.tokenizer.decode(input_ids_batch[i]))
                if self.mapping:
                    print("Content of self.mapping:", self.mapping)
                    label_list = self.mapping["label_list"]
                label_list = label_list.strip('][').split(', ')
                prediction = [(token, label_list[prediction]) for token, prediction in
                              zip(tokens, predictions[0].tolist())]
                inferences.append(prediction)
            logger.info("Model predicted: '%s'", prediction)

        return inferences, piece2word

    def useful_string(self, s):
        if s.istitle or s in string.punctuation:
            return False
        # punctuation、operator、build-in identifier
        non_var = ['@', '=', '!', '.', '+', '-', '*', '/', '++', '–', '==', '!=', '|', '||', '&', '&&', '+=', '-=',
                   '*=', '/=', '%=', '<<=', '>>=', '&=', 'ˆ=', '|=', '>', '<', '>=', '<=', ':', '%', 'ˆ', '?', '%', ' ',
                   '<<', '<<<', '>>', '>>>', '...', '−>', 'instanceof']
        if s in non_var:
            return False
        return True

    def postprocess(self, inference_output, piece2word):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        print("shape of inference_output:", np.array(inference_output).shape)
        sentence = inference_output[0]

        # complete_word = []
        # complete_word_label = []
        complete_prediction = []
        res = []
        underline = False
        pre = piece2word[1]
        identifier = sentence[1][0]
        pred = sentence[1][1]
        for i in range(2, len(sentence)-1):
            token = sentence[i][0]
            label = sentence[i][1]
            if piece2word[i] == pre:
                identifier += token
            else:
                if token == '_':
                    identifier += token
                    underline = True
                elif underline:
                    underline = False
                    identifier += token
                else:
                    # complete_word.append(identifier)
                    # complete_word_label.append(pred)
                    complete_prediction.append([identifier, pred])
                    if pred == "Logging":
                        if self.useful_string(identifier):
                            res.append(identifier)
                    identifier = token
                    pred = label
                pre = piece2word[i]
        logger.info("Complete word prediction:", complete_prediction)
        return [res]

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        data_preprocess = self.preprocess(data)

        if not self._is_explain():
            output, pieces2word = self.inference(data_preprocess)
            output = self.postprocess(output, pieces2word)
        else:
            output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), None, 'ms')
        return output
