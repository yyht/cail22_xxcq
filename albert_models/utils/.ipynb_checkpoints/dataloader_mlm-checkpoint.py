import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import itertools
from operator import is_not
from functools import partial
import re

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)

from collections import namedtuple
_DocSpan = namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])

def slide_window(all_doc_tokens, max_length, doc_stride, offset=32):
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_length - offset:
            length = max_length - offset
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)
    return doc_spans

class data_generator_mlm(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', 
                 link_symbol='_', task_dict={}):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        self.seg_token = seg_token
        self.sep_token = sep_token
        self.start_token = start_token
        self.link_symbol = link_symbol
        self.doc_stride = 32
        self.offset = 8
        self.task_dict = task_dict
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token, self.sep_token, self.start_token]})
        
        from utils.mlm_generator import MLMGenerator
        self.mlm_generator = MLMGenerator(
                                self.task_dict.get('mask_ratio', 0.15), 
                                self.task_dict.get('random_ratio', 0.1),
                                self.task_dict.get('min_tok', 2),
                                self.task_dict.get('max_tok', 10),
                                self.task_dict.get('mask_id', 103),
                                self.task_dict.get('pad', 0),
                                self.task_dict.get('geometric_p', 0.1),
                                self.tokenizer.get_vocab(),
                                self.task_dict.get('max_pair_targets', 72),
                                replacement_method='word_piece',
                                endpoints='')

        self.features = []
        for item in self.data:
            encoder_text = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoder_text['input_ids']
            text = item['text']
            
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
                
                content = {}
                
                content['span_start'] = span_start
                content['span_end'] = span_end
                content['input_ids'] = input_ids
                
                self.features.append(content)
        self.labels = [label] * len(self.features)
                
    def encoder(self, item):
        input_ids = item["input_ids"]
        
        span_start = item['span_start']
        span_end = item['span_end']

        span_input_ids = input_ids[span_start:span_end]
        [masked_sent, 
        masked_target, 
        _] = self.mlm_generator.ner_span_mask(
                        span_input_ids, 
                        self.tokenizer,
                        entity_spans=None,
                        return_only_spans=False,
                        ner_masking_prob=0.15,
                        mask_num=self.task_dict.get('max_pair_targets', len(span_input_ids)*self.task_dict.get('mask_ratio', 0.15))
                       )
        masked_sent = masked_sent.tolist()
        masked_target = masked_target.tolist()

        masked_sent = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + masked_sent
        masked_sent += self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        
        ori_sent = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + span_input_ids
        ori_sent += self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        
        masked_type_ids = [0] * len(masked_sent)
        masked_attention_mask = [1] * len(masked_sent)
        
        masked_target = [0]*len(self.tokenizer(self.start_token, add_special_tokens=False)['input_ids']) + masked_target + [0]*len(self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids'])
        return masked_sent, masked_type_ids, masked_attention_mask, masked_target, ori_sent
                
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = self.features[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_labels = []
        batch_ori_token_ids = []
        text_list = []
        for item in examples:
            masked_sent, masked_type_ids, masked_attention_mask, masked_target, ori_sent = item
            batch_token_ids.append(masked_sent)
            batch_mask_ids.append(masked_attention_mask)
            batch_token_type_ids.append(masked_type_ids)
            batch_labels.append(masked_target)
            batch_ori_token_ids.append(ori_sent)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_labels = torch.tensor(sequence_padding(batch_labels)).long()
        batch_ori_token_ids = torch.tensor(sequence_padding(batch_ori_token_ids)).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_labels, batch_ori_token_ids
    