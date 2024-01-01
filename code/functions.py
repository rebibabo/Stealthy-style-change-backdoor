import warnings
warnings.filterwarnings("ignore")
import argparse
import logging
import os
import pickle
import random
import json
import numpy as np
import torch
import time
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from model import Model
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

cpu_cont = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def process_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    config.num_labels=args.label_number
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    print("loading pre-train model...")
    model = model_class(config)
    model = Model(model,config,tokenizer,args)
    print("loading model from {}...".format(args.ori_model_path))
    model.load_state_dict(torch.load(args.ori_model_path))
    print("finish")
    model.to(device)
    return model, tokenizer

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=idx
        self.label=label
        
def convert_examples_to_features(code, label, tokenizer, args, i):
    #source
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,i,label)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        self.source = {}
        # To-Do: 这里需要根据code authorship的数据集重新做.
        file_type = '.'.join(file_path.split('/')[-1].split('.')[:-1])
        folder = '/'.join(file_path.split('/')[:-1]) # 得到文件目录

        cache_file_path = os.path.join(folder, 'cached_{}'.format(
                                    file_type))
        code_pairs_file_path = os.path.join(folder, 'cached_{}.pkl'.format(
                                    file_type))

        print('\n cached_features_file: ',cache_file_path)
        try:
            self.examples = torch.load(cache_file_path)
            with open(code_pairs_file_path, 'rb') as f:
                code_files = pickle.load(f)
            
            logger.info("Loading features from cached file %s", cache_file_path)
        
        except:
            logger.info("Creating features from dataset file at %s", file_path)
            code_files = []
            with open(file_path) as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    js = json.loads(line)
                    self.source[i] = js
                    code, label = js['code'], js['label']
                    self.examples.append(convert_examples_to_features(code, int(label), tokenizer, args, i))
                    code_files.append(code)
            assert(len(self.examples) == len(code_files))
            with open(code_pairs_file_path, 'wb') as f:
                pickle.dump(code_files, f)
            logger.info("Saving features into cached file %s", cache_file_path)
            torch.save(self.examples, cache_file_path)

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label),torch.tensor(self.examples[item].idx)


def load_and_cache_examples(args, tokenizer, evaluate=False,test=False,):
    dataset = TextDataset(tokenizer, args, file_path=args.test_data_file if test else (args.eval_data_file if evaluate else args.train_data_file),block_size=args.block_size,)
    return dataset
