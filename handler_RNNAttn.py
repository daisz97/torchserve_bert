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
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


def load_embedding_dict(vec_path):
    from tqdm import tqdm
    import numpy as np
    embeddings_index = dict()

    with open(vec_path, 'r', encoding='UTF-8') as file:
        for line in tqdm(file.readlines()):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    return embeddings_index


def read_all_data(path_to_file):  # max sequence length of snippet of this project
    words = []
    with open(path_to_file, 'r', encoding='utf-8') as file:  # max_len？是
        # sentence_len = 0
        for line in file:
            splitted = line.split()
            if len(splitted) == 0:
                # while sentence_len < max_len:
                #     words.append('<UNKNOWN>')
                #     sentence_len += 1
                # sentence_len = 0
                continue
            words.append(splitted[0])
            # sentence_len += 1
        return words


# Indexer是为了在embedding层将输入embedding转化为Glove embedding，即转化为词表的id
class Indexer:
    def __init__(self, elements):
        self._element_to_index = {"<UNKNOWN>": 0, "Logging": 1, "N-Logging": 2}

        for x in elements:
            if x not in self._element_to_index:
                self._element_to_index[x] = len(self._element_to_index)

        self._index_to_element = {v: k for k, v in self._element_to_index.items()}

    def get_element_to_index_dict(self):
        return self._element_to_index

    def element_to_index(self, element):
        return self._element_to_index.get(element, 0)

    def element_to_index_2d(self, element):
        return [self._element_to_index.get(x, 0) for x in element]

    def element_to_index_2d_min_1(self, element):  # 并减1
        return [self._element_to_index.get(x, 0)-1 for x in element]

    def index_to_element(self, index):
        return self._index_to_element[index]

    def elements_to_index(self, elements):
        return [self.element_to_index_2d(x) for x in elements]

    def elements_to_index_min_1(self, elements):
        return [self.element_to_index_2d_min_1(x) for x in elements]

    def indexes_to_elements(self, indexes):
        return [self.index_to_element(x) for x in indexes]

    def size(self):
        return len(self._element_to_index)


# 模型注册时参考：https://github.com/pytorch/serve/tree/master/examples/text_classification
class RNNAttnSeqClassifierHandler(BaseHandler, ABC):
    """
    RNN_Attn handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
       First try to load torchscript else load eager mode state_dict based model.
        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing
        """
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
            self.model.to(self.device)
        else:
            logger.debug("Loading torchscript model")
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")

            self.model = self._load_torchscript_model(model_pt_path)

        self.model.eval()

        logger.debug('Model file %s loaded successfully', model_pt_path)

        # Load class mapping for classifiers
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning("Missing the index_to_name.json file.")
        # train/dev/test path, glove embedding
        train_path = os.path.join(model_dir, "train.txt")
        dev_path = os.path.join(model_dir, "dev.txt")
        test_path = os.path.join(model_dir, "test.txt")
        embedding_path = os.path.join(model_dir, "glove.6B.100d.txt")

        logger.info("loading glove embedding......")
        glove = load_embedding_dict(embedding_path)
        logger.info("Done!")
        if os.path.isfile(train_path):
            train_words = read_all_data(train_path)
        else:
            raise RuntimeError("Missing the train.txt file")
        if os.path.isfile(dev_path):
            dev_words = read_all_data(dev_path)
        else:
            raise RuntimeError("Missing the dev.txt file")
        if os.path.isfile(test_path):
            test_words = read_all_data(test_path)
        else:
            raise RuntimeError("Missing the test.txt file")

        self.whole_words_indexer = Indexer(train_words + dev_words + test_words)

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
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode('utf-8')
            max_length = self.setup_config["max_length"]
            logger.info("Received text: '%s'", input_text)
            # preprocessing text for token_classification.
            # 1、filter string constant
            pattern = re.compile(r'\"(.*?)\"|\'(.*?)\'')  # 最短匹配：https://www.cnblogs.com/baxianhua/p/8571967.html
            input_text = re.sub(pattern, "", input_text)  # https://www.lidihuo.com/python/python-reg-expression.html
            print("input after string eliminated:", input_text)
            logger.info("input after string eliminated:'%s'", input_text)
            # 2、form input id

            input_text_token = word_tokenize(input_text)
            input_ids = torch.tensor(self.whole_words_indexer.elements_to_index(input_text_token), dtype=torch.long)
            input_ids_batch = input_ids.to(self.device)

            # inputs = self.tokenizer.encode_plus(input_text, max_length=int(max_length), pad_to_max_length=False,
            #                                     add_special_tokens=True, return_tensors='pt')  # pad_to_max_length=True
            #
            # input_ids = inputs["input_ids"].to(self.device)
            # piece2word = inputs.words()  # 值是从0开始，[CLS]和[SEP]是None
            # attention_mask = inputs["attention_mask"].to(self.device)
            # # making a batch out of the recieved requests
            # # attention masks are passed for cases where input tokens are padded.
            # if input_ids.shape is not None:
            #     if input_ids_batch is None:
            #         input_ids_batch = input_ids
            #         attention_mask_batch = attention_mask
            #     else:
            #         input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
            #         attention_mask_batch = torch.cat((attention_mask_batch, attention_mask), 0)
        # TO DO:增加超过512的前面截断处理
        return input_ids_batch

    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        input_ids_batch = input_batch
        inferences = []

        outputs = model(inputs)
        logger.info("The shape of output: '%s'", outputs.size())
        print("The shape of output:", outputs.size())
        print("output:", outputs)
        num_rows = outputs.shape[0]

        for i in range(num_rows):
            output = outputs[i].unsqueeze(0)
            predictions = torch.argmax(output, dim=2)
            tokens = self.whole_words_indexer.index_to_element(input_ids_batch[i])

            if self.mapping:
                print("Content of self.mapping:", self.mapping)
                label_list = self.mapping["label_list"]
            label_list = label_list.strip('][').split(', ')
            prediction = [(token, label_list[prediction]) for token, prediction in
                          zip(tokens, predictions[0].tolist())]
            inferences.append(prediction)
        logger.info("Model predicted: '%s'", prediction)

        return inferences

    def useful_string(self, s):
        if s.istitle() or s in string.punctuation:
            return False
        # punctuation、operator、build-in identifier
        non_var = ['@', '=', '!', '.', '+', '-', '*', '/', '++', '–', '==', '!=', '|', '||', '&', '&&', '+=', '-=',
                   '*=', '/=', '%=', '<<=', '>>=', '&=', 'ˆ=', '|=', '>', '<', '>=', '<=', ':', '%', 'ˆ', '?', '%', ' ',
                   '<<', '<<<', '>>', '>>>', '...', '−>', 'instanceof', 'boolean', 'int', 'long', 'short', 'byte',
                   'float', 'double', 'char', 'class', 'interface', 'if', 'else', 'do', 'while', 'for', 'switch',
                   'case', 'default', 'break', 'continue', 'return', 'try', 'catch', 'finally', 'public', 'protected',
                   'private', 'final', 'void', 'static', 'strict', 'abstract', 'transient', 'synchronized', 'volatile',
                   'native', 'package', 'import', 'throw', 'throws', 'extends', 'implements', 'this', 'supper', 'new',
                   'true', 'false', 'null', 'goto', 'const']
        if s in non_var:
            return False
        return True

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        print("shape of inference_output:", np.array(inference_output).shape)
        sentence = inference_output[0]  # 因为第一维代表batch的大小

        # complete_word = []
        # complete_word_label = []
        complete_prediction = []
        res = []
        identifier = sentence[1][0]
        pred = sentence[1][1]
        for i in range(2, len(sentence)-1):
            token = sentence[i][0]
            label = sentence[i][1]
            complete_prediction.append([token, label])
            if pred == "Logging":
                print(token)
                if self.useful_string(token):
                    res.append(token)

        logger.info("Complete word prediction:", complete_prediction)
        if len(res) == 0:
            res.append("No variable should be logged!")
        else:
            res = list(set(res))  # 对res去重
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
            output = self.inference(data_preprocess)
            output = self.postprocess(output)
        else:
            output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), None, 'ms')
        return output
