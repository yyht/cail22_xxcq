
import json
import logging
import sys
import functools
import random
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import itertools
from operator import is_not
from functools import partial
import re

def load_name(filename, valid_schema={}):
    #{"text": "62号汽车故障报告综合情况:故障现象:加速后，丢开油门，发动机熄火。", "spo_list": [{"predicate": "部件故障", "arguments": [{"role": "部件单元", "argument": "发动机", "argument_start_index": 28}, {"role": "故障状态", "argument": "熄火", "argument_start_index": 31}]}]}
    D = []
    print('===valid-schema===', valid_schema)
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            d = {'text': line['text'], 'relation': []}
            subj = set()
            obj = set()
            relation = {}
            for spo in line["spo_list"]:
                if not spo:
                    print(spo, '==invalid spo==')
                    continue

                start = spo['arguments'][0].get('argument_start_index', -1)
                end = len(spo['arguments'][0]['argument']) + start - 1
                subject_ = (start, end, spo['arguments'][0]['role'], spo['arguments'][0]['argument'])

                start = spo['arguments'][1].get('argument_start_index', -1)
                end = len(spo['arguments'][1]['argument']) + start - 1
                object_ = (start, end, spo['arguments'][1]['role'], spo['arguments'][1]['argument'])
                
#                 if object_[1] <= object_[0]:
#                     print(spo, '===inlvaid object===', object_)
                    
#                 if subject_[1] <= subject_[0]:
#                     print(spo, '===inlvaid subject===', subject_)

                subj.add(subject_)
                obj.add(object_)
                
                if subject_ + object_ not in relation:
                    relation[subject_ + object_] = subject_[2] + '-' + spo['predicate'] + '-' +object_[2]
                # else:
                #     print('======duplicate======')
                #     print(subject_ + object_, spo['predicate'], '=====', relation[subject_ + object_])

            pair_so = beam_search([subj, obj])
            for pair in pair_so:
                so = tuple(pair[0]) + tuple(pair[1])
                if so not in relation:
                    if valid_schema:
                        if (so[2], so[6]) in valid_schema:
                            relation[so] = 'null'
                    else:
                        relation[so] = 'null'
                so = tuple(pair[1]) + tuple(pair[0])
                if so not in relation:
                    if valid_schema:
                        if (so[2], so[6]) in valid_schema:
                            relation[so] = 'null'
                    else:
                        relation[so] = 'null'
            
            pair_so = beam_search([subj, subj])
            for pair in pair_so:
                if pair[0] == pair[1]:
                    continue
                so = tuple(pair[0]) + tuple(pair[1])
                if valid_schema:
                    if (so[2], so[6]) in valid_schema:
                        relation[so] = 'null'
                else:
                    relation[so] = 'null'
                        
            pair_so = beam_search([obj, obj])
            for pair in pair_so:
                if pair[0] == pair[1]:
                    continue
                so = tuple(pair[0]) + tuple(pair[1])
                if valid_schema:
                    if (so[2], so[6]) in valid_schema:
                        relation[so] = 'null'
                else:
                    relation[so] = 'null'
                    
            for key in relation:
                d['relation'].append((relation[key], )+key)
            
            if not d['relation']:
                continue
            D.append(d)
        random.shuffle(D)
        print(filename, '==size==', len(D))
        return D

def beam_search(dict_data):
    sequences = [list()]
    for item_list in dict_data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq = sequences[i]
            for item in item_list:
                candidate = seq+[item]
                all_candidates.append(candidate)
        sequences = all_candidates
    return sequences
    
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

def flat_list(h_list):
    e_list = []

    for item in h_list:
        if isinstance(item, list):
            e_list.extend(flat_list(item))
        else:
            e_list.append(item)
    return e_list

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def search_all(pattern, sequence):
    all_index = []
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            all_index.append(i)
    return all_index

def char_span_to_token_span(char2token, char_span):
    token_indexes = char2token[char_span[0]:char_span[1]]
    token_indexes = list(filter(partial(is_not, None), token_indexes))
    if token_indexes:
        return token_indexes[0], token_indexes[-1] + 1  # [start, end)
    else:  # empty
        return 0, 0

def token_span_to_char_span(token2char, token_span):
    char_indexes = token2char[token_span[0]:token_span[1]]
    char_indexes = [span for span in char_indexes]  # 删除CLS/SEP对应的span
    start, end = char_indexes[0][0], char_indexes[-1][1]
    return start, end

def get_token2char_char2token(tokenizer, text, maxlen):
    tokend = tokenizer(text, return_offsets_mapping=True, max_length=maxlen, truncation=True)
    token2char = tokend.offset_mapping
    
    char2token = [None] * len(text)
    for i, ((start, end)) in enumerate(token2char):
        char2token[start:end] = [i] * (end - start)
    
    return token2char, char2token

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

def add_marker_tokens(tokenizer, ner_labels):
    new_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>'%label)
        new_tokens.append('<SUBJ_END=%s>'%label)
        new_tokens.append('<OBJ_START=%s>'%label)
        new_tokens.append('<OBJ_END=%s>'%label)
    for label in ner_labels:
        new_tokens.append('<SUBJ=%s>'%label)
        new_tokens.append('<OBJ=%s>'%label)
    tokenizer.add_tokens(new_tokens)

class data_generator_slide_window(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, ner_labels=[], label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', link_symbol='_',
                mode='upsampling', unused_tokens=True, chunk_num=1, marker_type='entity_type'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        self.seg_token = seg_token
        self.sep_token = sep_token
        self.start_token = start_token
        self.link_symbol = link_symbol
        self.mode = mode
        self.doc_stride = 32
        self.offset = 8
        self.label = label
        self.chunk_num = chunk_num
        self.marker_type = marker_type
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token, self.sep_token, self.start_token]})
        
        add_marker_tokens(self.tokenizer, ner_labels)
        self.special_tokens = {}
        for w in ner_labels:
            if w not in self.special_tokens:
                if unused_tokens:
                    self.special_tokens[w] = "[unused%d]" % (len(self.special_tokens) + 1)
                else:
                    self.special_tokens[w] = ('<' + w + '>').lower()
        
        print(self.special_tokens, '===special_tokens===', ner_labels)
        
        self.features = []
        labels = []
        from collections import Counter
        raw_label_counter = Counter()
        
        for item in self.data:
            encoder_text = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)
            tokens = self.tokenizer.tokenize(item['text'])
            input_ids = encoder_text['input_ids']
            text = item['text']
            
            token2char = encoder_text.offset_mapping
            char2token = [None] * len(text)
            for i, ((start, end)) in enumerate(token2char):
                char2token[start:end] = [i] * (end - start)
            
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
                
                span_input_ids = input_ids[span_start:span_end]
                
                negative_relation = []
                        
                for relation in item['relation']:
                    
                    relation_label = relation[0]

                    valid_relation = []
                    subject_start, subject_end = relation[1], relation[2]
                    [subject_start_t, 
                     subject_end_t] = char_span_to_token_span(char2token, (subject_start, subject_end+1))

                    object_start, object_end = relation[5], relation[6]
                    [object_start_t, 
                     object_end_t] = char_span_to_token_span(char2token, (object_start, object_end+1))

                    subject_flag = subject_start_t >= span_start and subject_end_t <= span_end
                    object_flag = object_start_t >= span_start and object_end_t <= span_end
                    
                    value_flag = input_ids[subject_start_t:subject_end_t] and input_ids[object_start_t:object_end_t]

                    if not subject_flag or not object_flag or not value_flag:
                        continue
                        
                    content = {}
                    content['relation'] = {
                        'subj_end':subject_end_t-1-span_start,
                        'subj_start':subject_start_t-span_start,
                        'obj_end': object_end_t-1-span_start,
                        'obj_start': object_start_t-span_start,
                        'label':self.schema[relation_label],
                        'subj_type': relation[3],
                        'obj_type': relation[7],
                        'relation_label':relation_label
                    }
                    
                    # print(self.tokenizer.decode(span_input_ids[subject_start_t-span_start:subject_end_t-span_start]), 
                    #           '====', 
                    #           self.tokenizer.decode(span_input_ids[object_start_t-span_start:object_end_t-span_start]), 
                    #           '=====', relation)
                        
                    if relation_label not in ['null']:
                        for key in item:
                            if key == 'relation':
                                continue
                            else:
                                content[key] = item[key]
                        
                        content['span_start'] = span_start
                        content['span_end'] = span_end
                        content['input_tokens'] = tokens
                        labels.append(0)
                        self.features.append(content)
                        raw_label_counter[0] += 1
                    else:
                        negative_relation.append(content['relation'])
                        raw_label_counter[1] += 1
                        
                if negative_relation:
                    negative_relation_chunks = []
                    for x in range(0, len(negative_relation), self.chunk_num):
                        negative_relation_chunks.append(negative_relation[x:x+self.chunk_num])
                    for negative_relation_chunk in negative_relation_chunks:
                        neg_content = {}
                        for key in item:
                            if key == 'relation':
                                continue
                            else:
                                neg_content[key] = item[key]

                        neg_content['span_start'] = span_start
                        neg_content['span_end'] = span_end    
                        neg_content['input_tokens'] = tokens
                        neg_content['relation'] = negative_relation_chunk
                        self.features.append(neg_content)
                        labels.append(1)
        
        from collections import Counter
        label_counter = Counter()
        for label in labels:
            label_counter[label] += 1
        
        print('*****label_counter*****', label_counter, '***raw_label_counter***', raw_label_counter)
        
        import numpy as np
        labels = np.array(labels)
        samples_per_class = {
            label: (labels == label).sum() for label in set(labels)
        }

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(self.mode, str):
            assert self.mode in ["downsampling", "upsampling", 'nosampling']

        if isinstance(self.mode, int) or self.mode == "upsampling":
            samples_per_class = (
                self.mode
                if isinstance(self.mode, int)
                else max(samples_per_class.values())
            )
            print("** upsampling **")
        elif self.mode == "downsampling":
            samples_per_class = min(samples_per_class.values())
            print("** downsampling **")
        else:
            print('** nosampling **')

        # print("** samples_per_class **", samples_per_class)
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))
        
        self.indices = []
        for key in sorted(self.lbl2idx):
            replace_ = self.samples_per_class > len(self.lbl2idx[key])
            self.indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_
            ).tolist()
        assert len(self.indices) == self.length
        np.random.shuffle(self.indices)
        
        self.labels = [self.label] * len(self.indices)
        
        print('==total data==', len(self.indices), '==samples_per_class==', self.samples_per_class)
        
    def get_special_token(self, w, unused_tokens=True):
        if w not in self.special_tokens:
            if unused_tokens:
                self.special_tokens[w] = "[unused%d]" % (len(self.special_tokens) + 1)
            else:
                self.special_tokens[w] = ('<' + w + '>').lower()
        return self.special_tokens[w]
                
    def encoder(self, item):
        text = item["text"]
        input_tokens = item["input_tokens"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        if isinstance(item['relation'], list):
            import random
            random.shuffle(item['relation'])
            candidate_relation = item['relation'][0]
        else:
            candidate_relation = item['relation']
        
        span_input_tokens = input_tokens[span_start:span_end]
        
        if self.marker_type == 'entity_type':
            SUBJECT_START = self.get_special_token("SUBJ_START=%s"%candidate_relation['subj_type'])
            SUBJECT_END = self.get_special_token("SUBJ_END=%s"%candidate_relation['subj_type'])
            OBJECT_START = self.get_special_token("OBJ_START=%s"%candidate_relation['obj_type'])
            OBJECT_END = self.get_special_token("OBJ_END=%s"%candidate_relation['obj_type'])
        elif self.marker_type == 'entity_start_end':
            SUBJECT_START = self.get_special_token("SUBJ_START")
            SUBJECT_END = self.get_special_token("SUBJ_END")
            OBJECT_START = self.get_special_token("OBJ_START")
            OBJECT_END = self.get_special_token("OBJ_END")
        elif self.marker_type == 'entity':
            SUBJECT_START = self.get_special_token("SUBJ")
            SUBJECT_END = self.get_special_token("SUBJ")
            OBJECT_START = self.get_special_token("OBJ")
            OBJECT_END = self.get_special_token("OBJ")
        
        tokens = [self.start_token]
        for i, token in enumerate(span_input_tokens):
            if i == candidate_relation['subj_start']:
                sub_idx = len(tokens)
                tokens.append(SUBJECT_START)
            if i == candidate_relation['obj_start']:
                obj_idx = len(tokens)
                tokens.append(OBJECT_START)
            tokens.append(token)
            if i == candidate_relation['subj_end']:
                sub_end_idx = len(tokens)
                tokens.append(SUBJECT_END)
            if i == candidate_relation['obj_end']:
                obj_end_idx = len(tokens)
                tokens.append(OBJECT_END)

        tokens.append(self.sep_token)

        span_type_ids = [0] * len(tokens)        
        span_input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        span_attention_mask = [1] * len(span_input_ids)
        
        label_id = candidate_relation['label']
        
        # print(self.tokenizer.decode(span_input_ids), '====span_input_ids====', label_id, '===', candidate_relation['relation_label'])

        return [text, span_input_ids, span_attention_mask, span_type_ids, 
                label_id, sub_idx, obj_idx, sub_end_idx, obj_end_idx]
                
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        item = self.features[self.indices[idx]]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_labels, batch_sub_idx, batch_obj_idx = [], [], []
        batch_sub_end_idx, batch_obj_end_idx = [], []
        text_list = []
        for item in examples:
            [text, input_ids, attention_mask, token_type_ids, label, 
             sub_idx, obj_idx, sub_end_idx, obj_end_idx] = item
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(label)
            batch_sub_idx.append(sub_idx)
            batch_obj_idx.append(obj_idx)
            batch_sub_end_idx.append(sub_end_idx)
            batch_obj_end_idx.append(obj_end_idx)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_labels = torch.tensor(batch_labels).long()
        batch_sub_idx = torch.tensor(batch_sub_idx).long()
        batch_obj_idx = torch.tensor(batch_obj_idx).long()
        batch_sub_end_idx = torch.tensor(batch_sub_end_idx).long()
        batch_obj_end_idx = torch.tensor(batch_obj_end_idx).long()
        
        return [text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, 
                batch_labels, batch_sub_idx, batch_obj_idx,
               batch_sub_end_idx, batch_obj_end_idx]
    
    

class data_generator_slide_window_negative_sampling(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, ner_labels=[], label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', link_symbol='_',
                mode='upsampling', unused_tokens=True, chunk_num=10, marker_type='entity_type'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        self.seg_token = seg_token
        self.sep_token = sep_token
        self.start_token = start_token
        self.link_symbol = link_symbol
        self.mode = mode
        self.doc_stride = 32
        self.offset = 8
        self.label = label
        self.chunk_num = chunk_num
        self.marker_type = marker_type
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token, self.sep_token, self.start_token]})
        
        add_marker_tokens(self.tokenizer, ner_labels)
        self.special_tokens = {}
        for w in ner_labels:
            if w not in self.special_tokens:
                if unused_tokens:
                    self.special_tokens[w] = "[unused%d]" % (len(self.special_tokens) + 1)
                else:
                    self.special_tokens[w] = ('<' + w + '>').lower()
        
        print(self.special_tokens, '===special_tokens===', ner_labels)
        
        self.features = []
        labels = []
        from collections import Counter
        raw_label_counter = Counter()
        
        for item in self.data:
            encoder_text = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)
            tokens = self.tokenizer.tokenize(item['text'])
            input_ids = encoder_text['input_ids']
            text = item['text']
            
            token2char = encoder_text.offset_mapping
            char2token = [None] * len(text)
            for i, ((start, end)) in enumerate(token2char):
                char2token[start:end] = [i] * (end - start)
            
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
                
                span_input_ids = input_ids[span_start:span_end]
                
                negative_relation = []
                        
                for relation in item['relation']:
                    
                    relation_label = relation[0]

                    valid_relation = []
                    subject_start, subject_end = relation[1], relation[2]
                    [subject_start_t, 
                     subject_end_t] = char_span_to_token_span(char2token, (subject_start, subject_end+1))

                    object_start, object_end = relation[5], relation[6]
                    [object_start_t, 
                     object_end_t] = char_span_to_token_span(char2token, (object_start, object_end+1))

                    subject_flag = subject_start_t >= span_start and subject_end_t <= span_end
                    object_flag = object_start_t >= span_start and object_end_t <= span_end
                    
                    value_flag = input_ids[subject_start_t:subject_end_t] and input_ids[object_start_t:object_end_t]

                    if not subject_flag or not object_flag or not value_flag:
                        continue
                        
                    content = {}
                    content['relation'] = {
                        'subj_end':subject_end_t-1-span_start,
                        'subj_start':subject_start_t-span_start,
                        'obj_end': object_end_t-1-span_start,
                        'obj_start': object_start_t-span_start,
                        'label':self.schema[relation_label],
                        'subj_type': relation[3],
                        'obj_type': relation[7],
                        'relation_label':relation_label
                    }
                    
                    # print(self.tokenizer.decode(span_input_ids[subject_start_t-span_start:subject_end_t-span_start]), 
                    #           '====', 
                    #           self.tokenizer.decode(span_input_ids[object_start_t-span_start:object_end_t-span_start]), 
                    #           '=====', relation)
                        
                    if relation_label not in ['null']:
                        for key in item:
                            if key == 'relation':
                                continue
                            else:
                                content[key] = item[key]
                        
                        content['span_start'] = span_start
                        content['span_end'] = span_end
                        content['input_tokens'] = tokens
                        labels.append(0)
                        self.features.append(content)
                        raw_label_counter[0] += 1
                    else:
                        negative_relation.append(content['relation'])
                        raw_label_counter[1] += 1
                        
                if negative_relation:
                    negative_relation_chunks = []
                    for x in range(0, len(negative_relation), self.chunk_num):
                        negative_relation_chunks.append(negative_relation[x:x+self.chunk_num])
                    for _ in range(len(self.schema)-1):
                        for negative_relation_chunk in negative_relation_chunks:
                            neg_content = {}
                            for key in item:
                                if key == 'relation':
                                    continue
                                else:
                                    neg_content[key] = item[key]

                            neg_content['span_start'] = span_start
                            neg_content['span_end'] = span_end    
                            neg_content['input_tokens'] = tokens
                            neg_content['relation'] = negative_relation_chunk
                            self.features.append(neg_content)
                            labels.append(1)
        
        from collections import Counter
        label_counter = Counter()
        for label in labels:
            label_counter[label] += 1
        
        print('*****label_counter*****', label_counter, '***raw_label_counter***', raw_label_counter)
        
        import numpy as np
        labels = np.array(labels)
        samples_per_class = {
            label: (labels == label).sum() for label in set(labels)
        }

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(self.mode, str):
            assert self.mode in ["downsampling", "upsampling", 'nosampling']

        if isinstance(self.mode, int) or self.mode == "upsampling":
            samples_per_class = (
                self.mode
                if isinstance(self.mode, int)
                else max(samples_per_class.values())
            )
            print("** upsampling **")
        elif self.mode == "downsampling":
            samples_per_class = min(samples_per_class.values())
            print("** downsampling **")
        else:
            print('** nosampling **')

        # print("** samples_per_class **", samples_per_class)
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))
        
        self.indices = []
        for key in sorted(self.lbl2idx):
            replace_ = self.samples_per_class > len(self.lbl2idx[key])
            self.indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_
            ).tolist()
        assert len(self.indices) == self.length
        np.random.shuffle(self.indices)
        
        self.labels = [self.label] * len(self.indices)
        
        print('==total data==', len(self.indices), '==samples_per_class==', self.samples_per_class)
        
    def get_special_token(self, w, unused_tokens=True):
        if w not in self.special_tokens:
            if unused_tokens:
                self.special_tokens[w] = "[unused%d]" % (len(self.special_tokens) + 1)
            else:
                self.special_tokens[w] = ('<' + w + '>').lower()
        return self.special_tokens[w]
                
    def encoder(self, item):
        text = item["text"]
        input_tokens = item["input_tokens"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        if isinstance(item['relation'], list):
            import random
            random.shuffle(item['relation'])
            candidate_relation = item['relation'][0]
        else:
            candidate_relation = item['relation']
        
        span_input_tokens = input_tokens[span_start:span_end]
        
        if self.marker_type == 'entity_type':
            SUBJECT_START = self.get_special_token("SUBJ_START=%s"%candidate_relation['subj_type'])
            SUBJECT_END = self.get_special_token("SUBJ_END=%s"%candidate_relation['subj_type'])
            OBJECT_START = self.get_special_token("OBJ_START=%s"%candidate_relation['obj_type'])
            OBJECT_END = self.get_special_token("OBJ_END=%s"%candidate_relation['obj_type'])
        elif self.marker_type == 'entity_start_end':
            SUBJECT_START = self.get_special_token("SUBJ_START")
            SUBJECT_END = self.get_special_token("SUBJ_END")
            OBJECT_START = self.get_special_token("OBJ_START")
            OBJECT_END = self.get_special_token("OBJ_END")
        elif self.marker_type == 'entity':
            SUBJECT_START = self.get_special_token("SUBJ")
            SUBJECT_END = self.get_special_token("SUBJ")
            OBJECT_START = self.get_special_token("OBJ")
            OBJECT_END = self.get_special_token("OBJ")
            
        relation_label = candidate_relation['relation_label']
        label_id = 1
        if relation_label in ['null']:
            negative_label_list = list(set(self.schema.keys()) - set([relation_label]))
            random.shuffle(negative_label_list)
            relation_label = negative_label_list[0]
            label_id = 0
        
        tokens = [self.start_token] + self.tokenizer.tokenize(relation_label) + [self.sep_token]
        for i, token in enumerate(span_input_tokens):
            if i == candidate_relation['subj_start']:
                sub_idx = len(tokens)
                tokens.append(SUBJECT_START)
            if i == candidate_relation['obj_start']:
                obj_idx = len(tokens)
                tokens.append(OBJECT_START)
            tokens.append(token)
            if i == candidate_relation['subj_end']:
                sub_end_idx = len(tokens)
                tokens.append(SUBJECT_END)
            if i == candidate_relation['obj_end']:
                obj_end_idx = len(tokens)
                tokens.append(OBJECT_END)

        tokens.append(self.sep_token)

        span_type_ids = [0] * len(tokens)        
        span_input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        span_attention_mask = [1] * len(span_input_ids)
        
        # print(self.tokenizer.decode(span_input_ids), '====span_input_ids====', label_id, '===', candidate_relation['relation_label'])

        return [text, span_input_ids, span_attention_mask, span_type_ids, 
                label_id, sub_idx, obj_idx, sub_end_idx, obj_end_idx]
                
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        item = self.features[self.indices[idx]]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_labels, batch_sub_idx, batch_obj_idx = [], [], []
        batch_sub_end_idx, batch_obj_end_idx = [], []
        text_list = []
        for item in examples:
            [text, input_ids, attention_mask, token_type_ids, label, 
             sub_idx, obj_idx, sub_end_idx, obj_end_idx] = item
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(label)
            batch_sub_idx.append(sub_idx)
            batch_obj_idx.append(obj_idx)
            batch_sub_end_idx.append(sub_end_idx)
            batch_obj_end_idx.append(obj_end_idx)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_labels = torch.tensor(batch_labels).long()
        batch_sub_idx = torch.tensor(batch_sub_idx).long()
        batch_obj_idx = torch.tensor(batch_obj_idx).long()
        batch_sub_end_idx = torch.tensor(batch_sub_end_idx).long()
        batch_obj_end_idx = torch.tensor(batch_obj_end_idx).long()
        
        return [text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, 
                batch_labels, batch_sub_idx, batch_obj_idx,
               batch_sub_end_idx, batch_obj_end_idx]