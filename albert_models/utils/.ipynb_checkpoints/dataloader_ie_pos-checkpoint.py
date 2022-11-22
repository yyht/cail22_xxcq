import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import itertools
from operator import is_not
from functools import partial
import re

word_mapping = {
    '①':'1',
    '②':"2",
    '③':'3',
    '④':'4',
    '⑤':'5',
    '⑥':'6',
    '⑦':'7',
    '⑧':'8'
}

def load_test(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            d = {'text': line['text'], 'spoes': []}
            D.append(d)
    random.shuffle(D)
    return D

def load_name_split(filename):
    count = 0
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            flag = False
            text_list = re.split('[。;；]+', line['text'])
            for text in text_list:
                if not text:
                    continue
                sh = search(text, line['text'])
                st = sh + len(text)
                if sh < 0:
                    continue
                d = {'text': text, 'spoes': []}
                for spo in line["spo_list"]:
                    t = []
                    for a in spo['arguments']:
                        if a.get('argument_start_index', -1) < sh or a.get('argument_start_index', -1) > st:
                            continue
                        a_start = a.get('argument_start_index', -1) - sh
                        a_end = len(a['argument']) + a_start
                        if text[a_start:a_end] != a['argument']:
                            print(text[a_start:a_end], '===', a['argument'])
                            continue
                        t.append((
                            spo['predicate'], a['role'], a['argument'],
                            a.get('argument_start_index', -1) - sh
                        ))
                    if t:
                        d['spoes'].append(t)
                valid_list = [item for item in d['spoes'] if item and len(item) == 2]
                d['spoes'] = valid_list
                for item in d['spoes']:
                    if len(item) != 2:
                        count += 1
                D.append(d)
        random.shuffle(D)
        print(count)
        return D

# from itertools import permutations
def load_duee(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            d = {'text': line['text'], 'spoes': []}
            
            # for token in word_mapping:
            #     line['text'] = re.replace(token, word_mapping[token], line['text'])
            
            for e in line["event_list"]:
                for a in e['arguments']:
                    spo = []
                    if a:
                        spo.append((e['event_type'], '触发词', e['trigger'], e.get('trigger_start_index', -1)))
                        spo.append((e['event_type'], a['role'], a['argument'], a.get('argument_start_index', -1)))
                    if spo:
                        d['spoes'].append(spo)
            D.append(d)
        random.shuffle(D)
        return D

def load_name(filename):
    #{"text": "62号汽车故障报告综合情况:故障现象:加速后，丢开油门，发动机熄火。", "spo_list": [{"predicate": "部件故障", "arguments": [{"role": "部件单元", "argument": "发动机", "argument_start_index": 28}, {"role": "故障状态", "argument": "熄火", "argument_start_index": 31}]}]}
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            d = {'text': line['text'], 'spoes': []}
            for spo in line["spo_list"]:
                if spo:
                    d['spoes'].append([])
                for a in spo['arguments']:
                    d['spoes'][-1].append((
                        spo['predicate'], a['role'], a['argument'],
                        a.get('argument_start_index', -1)
                    ))
            D.append(d)
        print(D[0])
        random.shuffle(D)
        return D

def load_name_no_pos(filename):
    #{"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            d = {'text': line['text'], 'spoes': []}
            for spo in line["spo_list"]:
                if not spo:
                    continue
                for key in spo['object_type']:
                    if spo['subject'] and spo['object'][key]:
                        d['spoes'].append([])
                        d['spoes'][-1].append((
                            spo['predicate'], spo['subject_type'], spo['subject'],
                            -1
                        ))
                        d['spoes'][-1].append((
                            spo['predicate'], spo['object_type'][key], spo['object'][key],
                            -1
                        ))
            D.append(d)
        print(D[0])
        random.shuffle(D)
        return D
    
def load_name_no_pos_inverse(filename):
    #{"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            d = {'text': line['text'], 'spoes': []}
            for spo in line["spo_list"]:
                if not spo:
                    continue
                for key in spo['object_type']:
                    if spo['subject'] and spo['object'][key]:
                        d['spoes'].append([])
                        d['spoes'][-1].append((
                            spo['predicate'], spo['object_type'][key], spo['object'][key],
                            -1
                        ))
                        d['spoes'][-1].append((
                            spo['predicate'], spo['subject_type'], spo['subject'],
                            -1
                        ))
            D.append(d)
        print(D[0])
        random.shuffle(D)
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

import re
class data_generator(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, seg_token='<S>', 
                 link_symbol='_'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        self.labels = [label] * len(self.data)
        self.seg_token = seg_token
        self.link_symbol = link_symbol
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token] })
        
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        
        token2char = encoder_text.offset_mapping
        char2token = [None] * len(text)
        for i, ((start, end)) in enumerate(token2char):
            char2token[start:end] = [i] * (end - start)

        spoes = set()
        for spo in item['spoes']:
            p = self.schema[spo[0][1] + self.link_symbol + spo[0][0] + self.link_symbol + spo[1][1]]
            tmp_tuple = ()
            for (_, r, a, i) in spo:
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t]:
                        tmp_tuple += (start_t, end_t-1)
                        # print(r, a, self.tokenizer.decode(input_ids[start_t:end_t]))
                    
            if len(tmp_tuple) == 4:
                tmp_tuple += (p,)
                spoes.add(tmp_tuple)
        
        # if not spoes:
        #     print(item['spoes'], text)
        
        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        
        for sh, st, oh, ot, p in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])

        return text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels
    
class data_generator_slide_window(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', link_symbol='_'):
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
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token, self.sep_token, self.start_token]})
        
        self.features = []
        for item in self.data:
            encoder_text = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)
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
                
                content = {}
                
                for key in item:
                    content[key] = item[key]
                
                content['span_start'] = span_start
                content['span_end'] = span_end
                content['token2char'] = token2char
                content['char2token'] = char2token
                content['input_ids'] = input_ids
                
                self.features.append(content)
        self.labels = [label] * len(self.features)
                
    def encoder(self, item):
        text = item["text"]
        input_ids = item["input_ids"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        token2char = item['token2char']
        char2token = item['char2token']

        span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        candidate_start = 1

        spoes = set()
        for spo in item['spoes']:
            tmp_tuple = ()
            tmp_tuple_list = []
            p = spo[0][1] + self.link_symbol + spo[0][0] + self.link_symbol + spo[1][1]
            for (_, r, a, i) in spo:
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                        start_t_span = start_t - span_start + candidate_start
                        end_t_span = end_t - span_start + candidate_start
                        tmp_tuple += (start_t_span, end_t_span-1)
                        # print(r, a, self.tokenizer.decode(input_ids[start_t:end_t]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
                else:
                    a_token = self.tokenizer.encode(a, add_special_tokens=False)
                    a_list = search_all(a_token, input_ids[span_start:span_end])
                    if a_list:
                        tmp_tuple_list.append([])
                        for a_pos in a_list:
                            start_t_span = a_pos + candidate_start
                            end_t_span = a_pos + len(a_token) + candidate_start
                            tmp_tuple_list[-1].append((start_t_span, end_t_span-1))
                        
                            # import random
                            # if random.random() >= 0.1:
                            #     print(r, a, self.tokenizer.decode(input_ids[a_pos:a_pos+len(a_token)]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
 
            if len(tmp_tuple) == 4:
                tmp_tuple += (self.schema[p], )
                spoes.add(tmp_tuple)
            elif len(tmp_tuple_list) == 2:
                tuple_list = beam_search(tmp_tuple_list)
                for item_pair in tuple_list:
                    tmp_tuple_ = item_pair[0]
                    tmp_tuple_ += item_pair[1]
                    if len(tmp_tuple_) == 4:
                        tmp_tuple_ += (self.schema[p], )
                        spoes.add(tmp_tuple_)
        
        # if not spoes:
        #     print(item['spoes'], text)
        
        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        
        for sh, st, oh, ot, p in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])

        return text, entity_labels, head_labels, tail_labels, span_input_ids, span_attention_mask, span_type_ids
                
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
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels
    
class data_generator_slide_window_flatten(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', link_symbol='_'):
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
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token, self.sep_token, self.start_token]})
        
        self.features = []
        for item in self.data:
            encoder_text = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)
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
                
                content = {}
                
                for key in item:
                    content[key] = item[key]
                
                content['span_start'] = span_start
                content['span_end'] = span_end
                content['token2char'] = token2char
                content['char2token'] = char2token
                content['input_ids'] = input_ids
                
                self.features.append(content)
        self.labels = [label] * len(self.features)
                
    def encoder(self, item):
        text = item["text"]
        input_ids = item["input_ids"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        token2char = item['token2char']
        char2token = item['char2token']

        span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        candidate_start = 1

        spoes = set()
        for spo in item['spoes']:
            tmp_tuple = ()
            tmp_tuple_list = []
            p = spo[0][1] + self.link_symbol + spo[0][0] + self.link_symbol + spo[1][1]
            for (_, r, a, i) in spo:
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                        start_t_span = start_t - span_start + candidate_start
                        end_t_span = end_t - span_start + candidate_start
                        tmp_tuple += (start_t_span, end_t_span-1)
                        # print(r, a, self.tokenizer.decode(input_ids[start_t:end_t]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
                else:
                    a_token = self.tokenizer.encode(a, add_special_tokens=False)
                    a_list = search_all(a_token, input_ids[span_start:span_end])
                    if a_list:
                        tmp_tuple_list.append([])
                        for a_pos in a_list:
                            start_t_span = a_pos + candidate_start
                            end_t_span = a_pos + len(a_token) + candidate_start
                            tmp_tuple_list[-1].append((start_t_span, end_t_span-1))
                        
                            # import random
                            # if random.random() >= 0.1:
                            #     print(r, a, self.tokenizer.decode(input_ids[a_pos:a_pos+len(a_token)]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
 
            if len(tmp_tuple) == 4:
                tmp_tuple += (self.schema[p], )
                spoes.add(tmp_tuple)
            elif len(tmp_tuple_list) == 2:
                tuple_list = beam_search(tmp_tuple_list)
                for item_pair in tuple_list:
                    tmp_tuple_ = item_pair[0]
                    tmp_tuple_ += item_pair[1]
                    if len(tmp_tuple_) == 4:
                        tmp_tuple_ += (self.schema[p], )
                        spoes.add(tmp_tuple_)
        
        # if not spoes:
        #     print(item['spoes'], text)
        
        entity_labels = [set() for i in range(1)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        
        for sh, st, oh, ot, p in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[0].add((oh, ot))
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])

        return text, entity_labels, head_labels, tail_labels, span_input_ids, span_attention_mask, span_type_ids
                
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
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels
    

class data_generator_negative_learning_negative_balanced(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', link_symbol='-',
                 mode='upsampling', if_add_so=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        self.seg_token = seg_token
        self.sep_token = sep_token
        self.start_token = start_token
        self.doc_stride = 32
        self.offset = 8
        self.label = label
        from collections import Counter
        self.schema_count = Counter()
        self.link_symbol = link_symbol
        self.mode = mode
        self.if_add_so = if_add_so
        
        print(self.schema, '==schema==')

        self.schema_mapping = {}
        for idx, key in enumerate(['positive', 'negative']):
            self.schema_mapping[key] = idx
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token, self.sep_token, self.start_token]})
        
        self.features = []
        self.schema_features = {}
        labels = []
        positive_cnt = 0
        negative_cnt = 0
        for item in self.data:
            encoder_text = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoder_text['input_ids']
            text = item['text']
            
            predicate_dict = {}
            for spo in item['spoes']:
                p = spo[0][1] + self.link_symbol + spo[0][0] + self.link_symbol + spo[1][1]
                if p not in predicate_dict:
                    predicate_dict[p] = []
                predicate_dict[p].append(spo)
                
            current_target_type = set(list(predicate_dict.keys()))
            total_target_type = set(list(self.schema.keys()))

            left_target_type = list(total_target_type - current_target_type)
            
            token2char = encoder_text.offset_mapping
            char2token = [None] * len(text)
            for i, ((start, end)) in enumerate(token2char):
                char2token[start:end] = [i] * (end - start)
            
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
                
                for p in predicate_dict:
                    content = {}
                    for key in item:
                        if key in ['spoes']:
                            content[key] = predicate_dict[p]
                        else:
                            content[key] = item[key]
                        
                    content['span_start'] = span_start
                    content['span_end'] = span_end
                    content['token2char'] = token2char
                    content['char2token'] = char2token
                    content['input_ids'] = input_ids
                    
                    content['candidate_type'] = p
                    if p not in self.schema_features:
                        self.schema_features[p] = []
                    self.schema_features[p].append(content)
                    self.schema_count[p] += 1
                    positive_cnt += 1
                
                current_target_type = set(list(predicate_dict.keys()))
                total_target_type = set(list(self.schema.keys()))
                
                left_target_type = list(total_target_type - current_target_type)
                import random
                random.shuffle(left_target_type)
                
                if len(left_target_type) >= 1:
                    neg_content = {}
                    for key in item:
                        if key in ['spoes']:
                            neg_content[key] = []
                        else:
                            neg_content[key] = item[key]
                    neg_content['span_start'] = span_start
                    neg_content['span_end'] = span_end
                    neg_content['token2char'] = token2char
                    neg_content['char2token'] = char2token
                    neg_content['input_ids'] = input_ids
                    neg_content['candidate_type'] = left_target_type
                    self.features.append(neg_content)
                    labels.append(self.schema_mapping['negative'])
                    negative_cnt += 1

        import numpy as np

        cnt_list = [self.schema_count[key] for key in self.schema_count]
        median_cnt = np.median(cnt_list)

        print(median_cnt, '** median_cnt **')
                
        for p in self.schema_features:
            # if self.schema_count[p] < median_cnt:
            #     ratio = int(median_cnt/self.schema_count[p])
            # else:
            #     ratio = 1
            if self.schema_count[p] < 100:
                ratio = 5
            else:
                ratio = 1
            # print(p, '====', ratio)
            self.features.extend(self.schema_features[p]*ratio)
            labels.extend([self.schema_mapping['positive']]*len(self.schema_features[p])*ratio)
                      
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
        
        print('==positive count==', positive_cnt, '==negative count==', negative_cnt)
        print('==total data==', len(self.indices), '==samples_per_class==', self.samples_per_class)
        
    def encode_so(self, item):
        text = item["text"]
        input_ids = item["input_ids"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        token2char = item['token2char']
        char2token = item['char2token']
        
        import random
        if isinstance(item['candidate_type'], list):
            random.shuffle(item['candidate_type'])
            candidate_type = item['candidate_type'][0]
        else:
            candidate_type = item['candidate_type']
            
        # print(candidate_type, '=====', item['candidate_type'])
        candidate_start = 0
        candidate_type_ids = self.tokenizer(candidate_type, add_special_tokens=False)['input_ids']

        # [cls]candidate_type[sep]
        span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + candidate_type_ids + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        candidate_start = len(span_input_ids)
        
        span_input_ids += input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)

        soes = {}
        for spo in item['spoes']:
            tmp_tuple = ()
            tmp_tuple_list = []
            for (_, r, a, i) in spo:
                a_token = self.tokenizer.encode(a, add_special_tokens=False)
                a_list = search_all(a_token, input_ids[span_start:span_end])
                if r not in soes:
                    soes[r] = set()
                if a_list:
                    for a_pos in a_list:
                        start_t_span = a_pos + candidate_start
                        end_t_span = a_pos + len(a_token) + candidate_start
                        # print(r, a, self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
                        soes[r].add((start_t_span, end_t_span-1))
        return soes

    def encoder(self, item):
        text = item["text"]
        input_ids = item["input_ids"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        token2char = item['token2char']
        char2token = item['char2token']
        
        import random
        if isinstance(item['candidate_type'], list):
            random.shuffle(item['candidate_type'])
            item['spoes'] = []
            candidate_type = item['candidate_type'][0]
        else:
            candidate_type = item['candidate_type']
            
        # print(candidate_type, '=====', item['candidate_type'])
        candidate_start = 0
        candidate_type_ids = self.tokenizer(candidate_type, add_special_tokens=False)['input_ids']

        # [cls]candidate_type[sep]
        span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + candidate_type_ids + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        candidate_start = len(span_input_ids)
        
        span_input_ids += input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)

        spoes = set()
        for spo in item['spoes']:
            tmp_tuple = ()
            tmp_tuple_list = []
            for (_, r, a, i) in spo:
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                        start_t_span = start_t - span_start + candidate_start
                        end_t_span = end_t - span_start + candidate_start
                        tmp_tuple += (start_t_span, end_t_span-1)
                        # print(r, a, self.tokenizer.decode(input_ids[start_t:end_t]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
                else:
                    a_token = self.tokenizer.encode(a, add_special_tokens=False)
                    a_list = search_all(a_token, input_ids[span_start:span_end])
                    if a_list:
                        tmp_tuple_list.append([])
                        for a_pos in a_list:
                            start_t_span = a_pos + candidate_start
                            end_t_span = a_pos + len(a_token) + candidate_start
                            tmp_tuple_list[-1].append((start_t_span, end_t_span-1))
                        
                            # import random
                            # if random.random() >= 0.1:
                            #     print(r, a, self.tokenizer.decode(input_ids[a_pos:a_pos+len(a_token)]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
 
            if len(tmp_tuple) == 4:
                spoes.add(tmp_tuple)
            elif len(tmp_tuple_list) == 2:
                tuple_list = beam_search(tmp_tuple_list)
                for item_pair in tuple_list:
                    tmp_tuple_ = item_pair[0]
                    tmp_tuple_ += item_pair[1]
                    if len(tmp_tuple_) == 4:
                        spoes.add(tmp_tuple_)
        # if not spoes:
        #     print(item['spoes'], text)
        
        entity_labels = [set() for i in range(2)]
        head_labels, tail_labels = set(), set()
        
        if self.if_add_so:
            soes = self.encode_so(item)
            subject_type, predicate, object_type = candidate_type.split(self.link_symbol)

            for r in soes:
                if r == subject_type:
                    for sh, st in soes[r]:
                        entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
                elif r == object_type:
                    for oh, ot in soes[r]:
                        entity_labels[1].add((oh, ot)) #实体提取：2个类型，头实体or尾实体
        
        for sh, st, oh, ot in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels.add((sh, oh)) #类似TP-Linker
            tail_labels.add((st, ot))
        for label in entity_labels + [head_labels, tail_labels]:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(head_labels)])
        tail_labels = sequence_padding([list(tail_labels)])

        return text, entity_labels, head_labels, tail_labels, span_input_ids, span_attention_mask, span_type_ids 
                
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
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels

    
class data_generator_class_balanced_negative_learning(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', link_symbol='_',
                mode='nosampling', if_add_so=False):
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
        from collections import Counter
        self.schema_count = Counter()
        self.mode = mode
        self.if_add_so = if_add_so
        
        print(self.schema, '==schema==')
        
        self.schema_mapping = {}
        
        for idx, key in enumerate(list(self.schema.keys()) + ['null']):
            self.schema_mapping[key] = idx
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token, self.sep_token, self.start_token]})
        
        self.features = []
        labels = []
        positive_cnt = 0
        negative_cnt = 0
        for item in self.data:
            encoder_text = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoder_text['input_ids']
            text = item['text']
            
            predicate_dict = {}
            for spo in item['spoes']:
                p = spo[0][1] + self.link_symbol + spo[0][0] + self.link_symbol + spo[1][1]
                if p not in predicate_dict:
                    predicate_dict[p] = []
                predicate_dict[p].append(spo)
            
            token2char = encoder_text.offset_mapping
            char2token = [None] * len(text)
            for i, ((start, end)) in enumerate(token2char):
                char2token[start:end] = [i] * (end - start)
            
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
                
                for p in predicate_dict:
                    content = {}
                    for key in item:
                        if key in ['spoes']:
                            content[key] = predicate_dict[p]
                        else:
                            content[key] = item[key]

                    content['span_start'] = span_start
                    content['span_end'] = span_end
                    content['token2char'] = token2char
                    content['char2token'] = char2token
                    content['input_ids'] = input_ids
                    content['candidate_type'] = p
                    self.features.append(content)
                    labels.append(self.schema_mapping[p])
                    positive_cnt += 1
                
                current_target_type = set(list(predicate_dict.keys()))
                total_target_type = set(list(self.schema.keys()))
                
                left_target_type = list(total_target_type - current_target_type)
                import random
                random.shuffle(left_target_type)
                
                if len(left_target_type) >= 1:
                    neg_content = {}
                    for key in item:
                        if key in ['spoes']:
                            neg_content[key] = []
                        else:
                            neg_content[key] = item[key]
                    neg_content['span_start'] = span_start
                    neg_content['span_end'] = span_end
                    neg_content['token2char'] = token2char
                    neg_content['char2token'] = char2token
                    neg_content['input_ids'] = input_ids
                    neg_content['candidate_type'] = left_target_type
                    self.features.append(neg_content)
                    labels.append(self.schema_mapping['null'])
                    negative_cnt += 1
        
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

        self.labels = labels
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
        
        print('==positive count==', positive_cnt, '==negative count==', negative_cnt)
        print('==total data==', len(self.indices), '==samples_per_class==', self.samples_per_class)
        
    def encode_so(self, item):
        text = item["text"]
        input_ids = item["input_ids"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        token2char = item['token2char']
        char2token = item['char2token']
        
        import random
        if isinstance(item['candidate_type'], list):
            random.shuffle(item['candidate_type'])
            candidate_type = item['candidate_type'][0]
            item['spoes'] = []
        else:
            candidate_type = item['candidate_type']
            
        # print(candidate_type, '=====', item['candidate_type'])
        candidate_start = 0
        candidate_type_ids = self.tokenizer(candidate_type, add_special_tokens=False)['input_ids']

        # [cls]candidate_type[sep]
        span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + candidate_type_ids + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        candidate_start = len(span_input_ids)
        
        span_input_ids += input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)

        soes = {}
        for spo in item['spoes']:
            tmp_tuple = ()
            tmp_tuple_list = []
            for (_, r, a, i) in spo:
                a_token = self.tokenizer.encode(a, add_special_tokens=False)
                a_list = search_all(a_token, input_ids[span_start:span_end])
                if r not in soes:
                    soes[r] = set()
                if a_list:
                    for a_pos in a_list:
                        start_t_span = a_pos + candidate_start
                        end_t_span = a_pos + len(a_token) + candidate_start
                        # print(r, a, self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
                        soes[r].add((start_t_span, end_t_span-1))
        return soes
                    
    def encoder(self, item):
        text = item["text"]
        input_ids = item["input_ids"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        token2char = item['token2char']
        char2token = item['char2token']
        
        import random
        if isinstance(item['candidate_type'], list):
            random.shuffle(item['candidate_type'])
            item['spoes'] = []
            candidate_type = item['candidate_type'][0]
        else:
            candidate_type = item['candidate_type']
            
        # print(candidate_type, '=====', item['candidate_type'])
        candidate_start = 0
        candidate_type_ids = self.tokenizer(candidate_type, add_special_tokens=False)['input_ids']

        # [cls]candidate_type[sep]
        span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + candidate_type_ids + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        candidate_start = len(span_input_ids)
        
        span_input_ids += input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)

        spoes = set()
        for spo in item['spoes']:
            tmp_tuple = ()
            tmp_tuple_list = []
            for (_, r, a, i) in spo:
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                        start_t_span = start_t - span_start + candidate_start
                        end_t_span = end_t - span_start + candidate_start
                        tmp_tuple += (start_t_span, end_t_span-1)
                        # print(r, a, self.tokenizer.decode(input_ids[start_t:end_t]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
                else:
                    a_token = self.tokenizer.encode(a, add_special_tokens=False)
                    a_list = search_all(a_token, input_ids[span_start:span_end])
                    if a_list:
                        tmp_tuple_list.append([])
                        for a_pos in a_list:
                            start_t_span = a_pos + candidate_start
                            end_t_span = a_pos + len(a_token) + candidate_start
                            tmp_tuple_list[-1].append((start_t_span, end_t_span-1))
                        
                            # import random
                            # if random.random() >= 0.1:
                            #     print(r, a, self.tokenizer.decode(input_ids[a_pos:a_pos+len(a_token)]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
 
            if len(tmp_tuple) == 4:
                spoes.add(tmp_tuple)
            elif len(tmp_tuple_list) == 2:
                tuple_list = beam_search(tmp_tuple_list)
                for item_pair in tuple_list:
                    tmp_tuple_ = item_pair[0]
                    tmp_tuple_ += item_pair[1]
                    if len(tmp_tuple_) == 4:
                        spoes.add(tmp_tuple_)
        # if not spoes:
        #     print(item['spoes'], text)
        
        entity_labels = [set() for i in range(2)]
        head_labels, tail_labels = set(), set()
        
        if self.if_add_so:
            soes = self.encode_so(item)
            subject_type, predicate, object_type = candidate_type.split(self.link_symbol)

            for r in soes:
                if r == subject_type:
                    for sh, st in soes[r]:
                        entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
                elif r == object_type:
                    for oh, ot in soes[r]:
                        entity_labels[1].add((oh, ot)) #实体提取：2个类型，头实体or尾实体
        
        for sh, st, oh, ot in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels.add((sh, oh)) #类似TP-Linker
            tail_labels.add((st, ot))
        for label in entity_labels + [head_labels, tail_labels]:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(head_labels)])
        tail_labels = sequence_padding([list(tail_labels)])

        return text, entity_labels, head_labels, tail_labels, span_input_ids, span_attention_mask, span_type_ids 
                
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
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels
    
class data_generator_negative_learning(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', link_symbol='_',
                if_add_so=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        self.seg_token = seg_token
        self.sep_token = sep_token
        self.start_token = start_token
        self.link_symbol = link_symbol
        self.if_add_so = if_add_so
        self.doc_stride = 32
        self.offset = 8
        self.label = label
        from collections import Counter
        self.schema_count = Counter()
        
        print(self.schema, '==schema==')
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token, self.sep_token, self.start_token]})
        
        self.features = []
        self.schema_features = {}
        positive_cnt = 0
        negative_cnt = 0
        for item in self.data:
            encoder_text = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoder_text['input_ids']
            text = item['text']
            
            predicate_dict = {}
            for spo in item['spoes']:
                p = spo[0][1] + self.link_symbol + spo[0][0] + self.link_symbol + spo[1][1]
                if p not in predicate_dict:
                    predicate_dict[p] = []
                predicate_dict[p].append(spo)
            
            token2char = encoder_text.offset_mapping
            char2token = [None] * len(text)
            for i, ((start, end)) in enumerate(token2char):
                char2token[start:end] = [i] * (end - start)
            
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
                
                for p in predicate_dict:
                    content = {}
                    for key in item:
                        if key in ['spoes']:
                            content[key] = predicate_dict[p]
                        else:
                            content[key] = item[key]

                    content['span_start'] = span_start
                    content['span_end'] = span_end
                    content['token2char'] = token2char
                    content['char2token'] = char2token
                    content['input_ids'] = input_ids
                    content['candidate_type'] = p
                    if p not in self.schema_features:
                        self.schema_features[p] = []
                    self.schema_features[p].append(content)
                    self.schema_count[p] += 1
                    positive_cnt += 1
                
                current_target_type = set(list(predicate_dict.keys()))
                total_target_type = set(list(self.schema.keys()))
                
                left_target_type = list(total_target_type - current_target_type)
                import random
                random.shuffle(left_target_type)
                
                if len(left_target_type) >= 1:
                    neg_content = {}
                    for key in item:
                        if key in ['spoes']:
                            neg_content[key] = []
                        else:
                            neg_content[key] = item[key]
                    neg_content['span_start'] = span_start
                    neg_content['span_end'] = span_end
                    neg_content['token2char'] = token2char
                    neg_content['char2token'] = char2token
                    neg_content['input_ids'] = input_ids
                    neg_content['candidate_type'] = left_target_type
                    self.features.append(neg_content)
                    negative_cnt += 1
                    
        print(self.schema_count, '==schema_count==')
        
        cnt_list = [self.schema_count[key] for key in self.schema_count]
        median_cnt = np.median(cnt_list)
                
        for p in self.schema_features:
            if self.schema_count[p] < median_cnt:
                ratio = int(median_cnt/self.schema_count[p])
            else:
                ratio = 1
            print(p, '====', ratio)
            self.features.extend(self.schema_features[p]*ratio)
                      
        self.labels = [self.label] * len(self.features)
        import random
        random.shuffle(self.features)
        
        print('==positive count==', positive_cnt, '==negative count==', negative_cnt)
        print('==total data==', len(self.features))
        
    def encode_so(self, item):
        text = item["text"]
        input_ids = item["input_ids"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        token2char = item['token2char']
        char2token = item['char2token']
        
        import random
        if isinstance(item['candidate_type'], list):
            random.shuffle(item['candidate_type'])
            item['spoes'] = []
            candidate_type = item['candidate_type'][0]
        else:
            candidate_type = item['candidate_type']
            
        # print(candidate_type, '=====', item['candidate_type'])
        candidate_start = 0
        candidate_type_ids = self.tokenizer(candidate_type, add_special_tokens=False)['input_ids']

        # [cls]candidate_type[sep]
        span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + candidate_type_ids + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        candidate_start = len(span_input_ids)
        
        span_input_ids += input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)

        soes = {}
        for spo in item['spoes']:
            tmp_tuple = ()
            tmp_tuple_list = []
            for (_, r, a, i) in spo:
                a_token = self.tokenizer.encode(a, add_special_tokens=False)
                a_list = search_all(a_token, input_ids[span_start:span_end])
                if r not in soes:
                    soes[r] = set()
                if a_list:
                    for a_pos in a_list:
                        start_t_span = a_pos + candidate_start
                        end_t_span = a_pos + len(a_token) + candidate_start
                        # print(r, a, self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
                        soes[r].add((start_t_span, end_t_span-1))
        return soes
        
    def encoder(self, item):
        text = item["text"]
        input_ids = item["input_ids"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        token2char = item['token2char']
        char2token = item['char2token']
        
        import random
        if isinstance(item['candidate_type'], list):
            random.shuffle(item['candidate_type'])
            item['spoes'] = []
            candidate_type = item['candidate_type'][0]
        else:
            candidate_type = item['candidate_type']
            
        # print(candidate_type, '=====', item['candidate_type'])
        candidate_start = 0
        candidate_type_ids = self.tokenizer(candidate_type, add_special_tokens=False)['input_ids']

        # [cls]candidate_type[sep]
        span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + candidate_type_ids + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        candidate_start = len(span_input_ids)
        
        span_input_ids += input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)

        spoes = set()
        for spo in item['spoes']:
            tmp_tuple = ()
            tmp_tuple_list = []
            for (_, r, a, i) in spo:
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                        start_t_span = start_t - span_start + candidate_start
                        end_t_span = end_t - span_start + candidate_start
                        tmp_tuple += (start_t_span, end_t_span-1)
                        # print(r, a, self.tokenizer.decode(input_ids[start_t:end_t]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
                else:
                    a_token = self.tokenizer.encode(a, add_special_tokens=False)
                    a_list = search_all(a_token, input_ids[span_start:span_end])
                    if a_list:
                        tmp_tuple_list.append([])
                        for a_pos in a_list:
                            start_t_span = a_pos + candidate_start
                            end_t_span = a_pos + len(a_token) + candidate_start
                            tmp_tuple_list[-1].append((start_t_span, end_t_span-1))
                        
                            # import random
                            # if random.random() >= 0.1:
                            #     print(r, a, self.tokenizer.decode(input_ids[a_pos:a_pos+len(a_token)]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
 
            if len(tmp_tuple) == 4:
                spoes.add(tmp_tuple)
            elif len(tmp_tuple_list) == 2:
                tuple_list = beam_search(tmp_tuple_list)
                for item_pair in tuple_list:
                    tmp_tuple_ = item_pair[0]
                    tmp_tuple_ += item_pair[1]
                    if len(tmp_tuple_) == 4:
                        spoes.add(tmp_tuple_)
        # if not spoes:
        #     print(item['spoes'], text)
        
        entity_labels = [set() for i in range(2)]
        head_labels, tail_labels = set(), set()
        
        if self.if_add_so:
            soes = self.encode_so(item)
            subject_type, predicate, object_type = candidate_type.split(self.link_symbol)

            for r in soes:
                if r == subject_type:
                    for sh, st in soes[r]:
                        entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
                elif r == object_type:
                    for oh, ot in soes[r]:
                        entity_labels[1].add((oh, ot)) #实体提取：2个类型，头实体or尾实体
        
        for sh, st, oh, ot in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels.add((sh, oh)) #类似TP-Linker
            tail_labels.add((st, ot))
        for label in entity_labels + [head_labels, tail_labels]:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(head_labels)])
        tail_labels = sequence_padding([list(tail_labels)])

        return text, entity_labels, head_labels, tail_labels, span_input_ids, span_attention_mask, span_type_ids 
                
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
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels
    
class data_generator_no_negative_learning(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', link_symbol='_'):
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
        from collections import Counter
        self.schema_count = Counter()
        
        print(self.schema, '==schema==')
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token, self.sep_token, self.start_token]})
        
        self.features = []
        self.schema_features = {}
        positive_cnt = 0
        negative_cnt = 0
        for item in self.data:
            encoder_text = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoder_text['input_ids']
            text = item['text']
            
            predicate_dict = {}
            for spo in item['spoes']:
                p = spo[0][1] + self.link_symbol + spo[0][0] + self.link_symbol + spo[1][1]
                if p not in predicate_dict:
                    predicate_dict[p] = []
                predicate_dict[p].append(spo)
            
            token2char = encoder_text.offset_mapping
            char2token = [None] * len(text)
            for i, ((start, end)) in enumerate(token2char):
                char2token[start:end] = [i] * (end - start)
            
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
                
                for p in predicate_dict:
                    content = {}
                    for key in item:
                        if key in ['spoes']:
                            content[key] = predicate_dict[p]
                        else:
                            content[key] = item[key]

                    content['span_start'] = span_start
                    content['span_end'] = span_end
                    content['token2char'] = token2char
                    content['char2token'] = char2token
                    content['input_ids'] = input_ids
                    content['candidate_type'] = p
                    if p not in self.schema_features:
                        self.schema_features[p] = []
                    self.schema_features[p].append(content)
                    self.schema_count[p] += 1
                    positive_cnt += 1
                    
        print(self.schema_count, '==schema_count==')
        
        cnt_list = [self.schema_count[key] for key in self.schema_count]
        median_cnt = np.median(cnt_list)
                
        for p in self.schema_features:
            if self.schema_count[p] < median_cnt:
                ratio = int(median_cnt/self.schema_count[p])
            else:
                ratio = 1
            print(p, '====', ratio)
            self.features.extend(self.schema_features[p]*ratio)
                      
        self.labels = [label] * len(self.features)
        import random
        random.shuffle(self.features)
        
        print('==positive count==', positive_cnt, '==negative count==', negative_cnt)
        print('==total data==', len(self.features))
                    
    def encoder(self, item):
        text = item["text"]
        input_ids = item["input_ids"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        token2char = item['token2char']
        char2token = item['char2token']
        
        import random
        if isinstance(item['candidate_type'], list):
            random.shuffle(item['candidate_type'])
            item['spoes'] = []
            candidate_type = item['candidate_type'][0]
        else:
            candidate_type = item['candidate_type']
            
        # print(candidate_type, '=====', item['candidate_type'])
        candidate_start = 0
        candidate_type_ids = self.tokenizer(candidate_type, add_special_tokens=False)['input_ids']

        # [cls]candidate_type[sep]
        span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + candidate_type_ids + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        candidate_start = len(span_input_ids)
        
        span_input_ids += input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)

        spoes = set()
        for spo in item['spoes']:
            tmp_tuple = ()
            tmp_tuple_list = []
            for (_, r, a, i) in spo:
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                        start_t_span = start_t - span_start + candidate_start
                        end_t_span = end_t - span_start + candidate_start
                        tmp_tuple += (start_t_span, end_t_span-1)
                        # print(r, a, self.tokenizer.decode(input_ids[start_t:end_t]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
                else:
                    a_token = self.tokenizer.encode(a, add_special_tokens=False)
                    a_list = search_all(a_token, input_ids[span_start:span_end])
                    if a_list:
                        tmp_tuple_list.append([])
                        for a_pos in a_list:
                            start_t_span = a_pos + candidate_start
                            end_t_span = a_pos + len(a_token) + candidate_start
                            tmp_tuple_list[-1].append((start_t_span, end_t_span-1))
                        
                            # import random
                            # if random.random() >= 0.1:
                            #     print(r, a, self.tokenizer.decode(input_ids[a_pos:a_pos+len(a_token)]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
 
            if len(tmp_tuple) == 4:
                spoes.add(tmp_tuple)
            elif len(tmp_tuple_list) == 2:
                tuple_list = beam_search(tmp_tuple_list)
                for item_pair in tuple_list:
                    tmp_tuple_ = item_pair[0]
                    tmp_tuple_ += item_pair[1]
                    if len(tmp_tuple_) == 4:
                        spoes.add(tmp_tuple_)
        # if not spoes:
        #     print(item['spoes'], text)
        
        entity_labels = [set() for i in range(2)]
        head_labels, tail_labels = set(), set()
        
        for sh, st, oh, ot in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels.add((sh, oh)) #类似TP-Linker
            tail_labels.add((st, ot))
        for label in entity_labels + [head_labels, tail_labels]:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(head_labels)])
        tail_labels = sequence_padding([list(tail_labels)])

        return text, entity_labels, head_labels, tail_labels, span_input_ids, span_attention_mask, span_type_ids 
                
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
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels
    
class data_generator_negative_learning_flatten(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', link_symbol='_'):
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
        from collections import Counter
        self.schema_count = Counter()
        
        print(self.schema, '==schema==')
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token, self.sep_token, self.start_token]})
        
        self.features = []
        self.schema_features = {}
        positive_cnt = 0
        negative_cnt = 0
        for item in self.data:
            encoder_text = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoder_text['input_ids']
            text = item['text']
            
            predicate_dict = {}
            for spo in item['spoes']:
                p = spo[0][1] + self.link_symbol + spo[0][0] + self.link_symbol + spo[1][1]
                if p not in predicate_dict:
                    predicate_dict[p] = []
                predicate_dict[p].append(spo)
            
            token2char = encoder_text.offset_mapping
            char2token = [None] * len(text)
            for i, ((start, end)) in enumerate(token2char):
                char2token[start:end] = [i] * (end - start)
            
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
                
                for p in predicate_dict:
                    content = {}
                    for key in item:
                        if key in ['spoes']:
                            content[key] = predicate_dict[p]
                        else:
                            content[key] = item[key]

                    content['span_start'] = span_start
                    content['span_end'] = span_end
                    content['token2char'] = token2char
                    content['char2token'] = char2token
                    content['input_ids'] = input_ids
                    content['candidate_type'] = p
                    if p not in self.schema_features:
                        self.schema_features[p] = []
                    self.schema_features[p].append(content)
                    self.schema_count[p] += 1
                    positive_cnt += 1
                
                current_target_type = set(list(predicate_dict.keys()))
                total_target_type = set(list(self.schema.keys()))
                
                left_target_type = list(total_target_type - current_target_type)
                import random
                random.shuffle(left_target_type)
                
                if len(left_target_type) >= 1:
                    neg_content = {}
                    for key in item:
                        if key in ['spoes']:
                            neg_content[key] = []
                        else:
                            neg_content[key] = item[key]
                    neg_content['span_start'] = span_start
                    neg_content['span_end'] = span_end
                    neg_content['token2char'] = token2char
                    neg_content['char2token'] = char2token
                    neg_content['input_ids'] = input_ids
                    neg_content['candidate_type'] = left_target_type
                    self.features.append(neg_content)
                    negative_cnt += 1
                    
        print(self.schema_count, '==schema_count==')
        
        for p in self.schema_features:
            if self.schema_count[p] <= 500:
                ratio = 5
            else:
                ratio = 1
            self.features.extend(self.schema_features[p]*ratio)
                      
        self.labels = [label] * len(self.features)
        import random
        random.shuffle(self.features)
        
        print('==positive count==', positive_cnt, '==negative count==', negative_cnt)
        print('==total data==', len(self.features))
                    
    def encoder(self, item):
        text = item["text"]
        input_ids = item["input_ids"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        token2char = item['token2char']
        char2token = item['char2token']
        
        import random
        if isinstance(item['candidate_type'], list):
            random.shuffle(item['candidate_type'])
            item['spoes'] = []
            candidate_type = item['candidate_type'][0]
        else:
            candidate_type = item['candidate_type']
            
        # print(candidate_type, '=====', item['candidate_type'])
        candidate_start = 0
        candidate_type_ids = self.tokenizer(candidate_type, add_special_tokens=False)['input_ids']

        # [cls]candidate_type[sep]
        span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + candidate_type_ids + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        candidate_start = len(span_input_ids)
        
        span_input_ids += input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)

        spoes = set()
        for spo in item['spoes']:
            tmp_tuple = ()
            tmp_tuple_list = []
            for (_, r, a, i) in spo:
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                        start_t_span = start_t - span_start + candidate_start
                        end_t_span = end_t - span_start + candidate_start
                        tmp_tuple += (start_t_span, end_t_span-1)
                        # print(r, a, self.tokenizer.decode(input_ids[start_t:end_t]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
                else:
                    a_token = self.tokenizer.encode(a, add_special_tokens=False)
                    a_list = search_all(a_token, input_ids[span_start:span_end])
                    if a_list:
                        tmp_tuple_list.append([])
                        for a_pos in a_list:
                            start_t_span = a_pos + candidate_start
                            end_t_span = a_pos + len(a_token) + candidate_start
                            tmp_tuple_list[-1].append((start_t_span, end_t_span-1))
                        
                            import random
                            if random.random() >= 0.1:
                                print(r, a, self.tokenizer.decode(input_ids[a_pos:a_pos+len(a_token)]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
 
            if len(tmp_tuple) == 4:
                spoes.add(tmp_tuple)
            elif len(tmp_tuple_list) == 2:
                tuple_list = beam_search(tmp_tuple_list)
                for item_pair in tuple_list:
                    tmp_tuple_ = item_pair[0]
                    tmp_tuple_ += item_pair[1]
                    if len(tmp_tuple_) == 4:
                        spoes.add(tmp_tuple_)
        # if not spoes:
        #     print(item['spoes'], text)
        
        entity_labels = [set() for i in range(1)]
        head_labels, tail_labels = set(), set()
        
        for sh, st, oh, ot in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[0].add((oh, ot))
            head_labels.add((sh, oh)) #类似TP-Linker
            tail_labels.add((st, ot))
        for label in entity_labels + [head_labels, tail_labels]:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(head_labels)])
        tail_labels = sequence_padding([list(tail_labels)])

        return text, entity_labels, head_labels, tail_labels, span_input_ids, span_attention_mask, span_type_ids 
                
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
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels
    
from itertools import combinations, permutations
class data_generator_negative_learning_query_mrc(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', link_symbol='_'):
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
        from collections import Counter
        self.schema_count = Counter()
        
        print(self.schema, '==schema==')
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token, self.sep_token, self.start_token]})
        
        self.features = []
        self.schema_features = {}
        positive_cnt = 0
        negative_cnt = 0
        for item in self.data:
            encoder_text = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoder_text['input_ids']
            text = item['text']
            
            predicate_dict = {}
            for spo in item['spoes']:
                p = spo[0][1] + self.link_symbol + spo[0][0] + self.link_symbol + spo[1][1]
                if p not in predicate_dict:
                    predicate_dict[p] = []
                predicate_dict[p].append(spo)
            
            token2char = encoder_text.offset_mapping
            char2token = [None] * len(text)
            for i, ((start, end)) in enumerate(token2char):
                char2token[start:end] = [i] * (end - start)
            
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
                
                current_target_type = set(list(predicate_dict.keys()))
                total_target_type = set(list(self.schema.keys()))
                
                left_target_type = list(total_target_type - current_target_type)
                import random
                random.shuffle(left_target_type)
                
                for p in predicate_dict:
                    for spo in predicate_dict[p]:
                        spo_combination_list = list(permutations(spo, 2))
                        # print(spo, '====', spo_combination_list)
                        for spo_combination in spo_combination_list:
                            content = {}
                            for key in item:
                                if key in ['spoes']:
                                    continue
                                else:
                                    content[key] = item[key]
                            content['query'] = spo_combination[0]
                            content['target'] = spo_combination[1]
                            content['span_start'] = span_start
                            content['span_end'] = span_end
                            content['token2char'] = token2char
                            content['char2token'] = char2token
                            content['input_ids'] = input_ids
                            content['candidate_type'] = p
                            if list(spo) == list(spo_combination):
                                content['label'] = 1
                            else:
                                content['label'] = 0
                            self.features.append(content)
                      
        self.labels = [label] * len(self.features)
        import random
        random.shuffle(self.features)
        print('==total data of query mrc==', len(self.features))
                    
    def encoder(self, item):
        text = item["text"]
        input_ids = item["input_ids"]
        
        query = item["query"]
        target = item["target"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        token2char = item['token2char']
        char2token = item['char2token']
        
        import random
        if isinstance(item['candidate_type'], list):
            random.shuffle(item['candidate_type'])
            candidate_type = item['candidate_type'][0]
        else:
            candidate_type = item['candidate_type']
            
        candidate_start = 0
        # query
        candidate_type_ids = self.tokenizer(query[2], add_special_tokens=False)['input_ids']
        # query [sep] schema-type
        candidate_type_ids += self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids'] + self.tokenizer(candidate_type, add_special_tokens=False)['input_ids']

        # [cls]query [sep] schema-type [sep]
        span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + candidate_type_ids + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        candidate_start = len(span_input_ids)
        
        # [cls]query [sep] schema-type [sep] input [sep]
        span_input_ids += input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        
        (_, r, a, i) = query
        label = 0
        if i != -1:
            start, end = i, i + len(a) - 1
            start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
            if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                label = 1
        else:
            a_token = self.tokenizer.encode(a, add_special_tokens=False)
            a_list = search_all(a_token, input_ids[span_start:span_end])
            if a_list:
                label = 1
        
        target_span_list = []
        if label == 1 and item['label'] == 1:
            (_, r, a, i) = target
            if i != -1:
                start, end = i, i + len(a) - 1
                start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                    start_t_span = start_t - span_start + candidate_start
                    end_t_span = end_t - span_start + candidate_start
                    target_span_list.append((start_t_span, end_t_span-1))
            else:
                a_token = self.tokenizer.encode(a, add_special_tokens=False)
                a_list = search_all(a_token, input_ids[span_start:span_end])
                if a_list:
                    for a_pos in a_list:
                        start_t_span = a_pos + candidate_start
                        end_t_span = a_pos + len(a_token) + candidate_start
                        target_span_list.append((start_t_span, end_t_span-1))
        
        entity_labels = [set() for i in range(1)]
        
        if label == 1 and item['label'] == 1:
            for sh, st in target_span_list:
                entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
        for label in entity_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])

        return text, entity_labels, span_input_ids, span_attention_mask, span_type_ids 
                
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
        batch_entity_labels = []
        text_list = []
        for item in examples:
            text, entity_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels
    
    
class data_generator_negative_learning_query(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', link_symbol='_'):
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
        from collections import Counter
        self.schema_count = Counter()
        
        print(self.schema, '==schema==')
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token, self.sep_token, self.start_token]})
        
        self.features = []
        self.schema_features = {}
        positive_cnt = 0
        negative_cnt = 0
        for item in self.data:
            encoder_text = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)
            input_ids = encoder_text['input_ids']
            text = item['text']
            
            predicate_dict = {}
            for spo in item['spoes']:
                p = spo[0][1] + self.link_symbol + spo[0][0] + self.link_symbol + spo[1][1]
                if p not in predicate_dict:
                    predicate_dict[p] = []
                predicate_dict[p].append(spo)
            
            token2char = encoder_text.offset_mapping
            char2token = [None] * len(text)
            for i, ((start, end)) in enumerate(token2char):
                char2token[start:end] = [i] * (end - start)
            
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
                
                for p in predicate_dict:
                    content = {}
                    for key in item:
                        if key in ['spoes']:
                            content[key] = predicate_dict[p]
                        else:
                            content[key] = item[key]

                    content['span_start'] = span_start
                    content['span_end'] = span_end
                    content['token2char'] = token2char
                    content['char2token'] = char2token
                    content['input_ids'] = input_ids
                    content['candidate_type'] = p
                    if p not in self.schema_features:
                        self.schema_features[p] = []
                    self.schema_features[p].append(content)
                    self.schema_count[p] += 1
                    positive_cnt += 1
                
                current_target_type = set(list(predicate_dict.keys()))
                total_target_type = set(list(self.schema.keys()))
                
                left_target_type = list(total_target_type - current_target_type)
                import random
                random.shuffle(left_target_type)
                
                if len(left_target_type) >= 1:
                    neg_content = {}
                    for key in item:
                        if key in ['spoes']:
                            neg_content[key] = []
                        else:
                            neg_content[key] = item[key]
                    neg_content['span_start'] = span_start
                    neg_content['span_end'] = span_end
                    neg_content['token2char'] = token2char
                    neg_content['char2token'] = char2token
                    neg_content['input_ids'] = input_ids
                    neg_content['candidate_type'] = left_target_type
                    self.features.append(neg_content)
                    negative_cnt += 1
                    
        print(self.schema_count, '==schema_count==')
        
        for p in self.schema_features:
            if self.schema_count[p] <= 500:
                ratio = 5
            else:
                ratio = 1
            self.features.extend(self.schema_features[p]*ratio)
                      
        self.labels = [label] * len(self.features)
        import random
        random.shuffle(self.features)
        
        print('==positive count==', positive_cnt, '==negative count==', negative_cnt)
        print('==total data of query==', len(self.features))
                    
    def encoder(self, item):
        text = item["text"]
        input_ids = item["input_ids"]
        
        span_start = item['span_start']
        span_end = item['span_end']
        
        token2char = item['token2char']
        char2token = item['char2token']
        
        import random
        if isinstance(item['candidate_type'], list):
            random.shuffle(item['candidate_type'])
            item['spoes'] = []
            candidate_type = item['candidate_type'][0]
        else:
            candidate_type = item['candidate_type']
            
        # print(candidate_type, '=====', item['candidate_type'])
        candidate_start = 0
        candidate_type_ids = self.tokenizer(candidate_type, add_special_tokens=False)['input_ids']

        # [cls]candidate_type[sep]
        span_input_ids = self.tokenizer(self.start_token, add_special_tokens=False)['input_ids'] + candidate_type_ids + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        candidate_start = len(span_input_ids)
        
        span_input_ids += input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)

        spoes = set()
        tmp_tuple_list = []
        for spo in item['spoes']:
            for (_, r, a, i) in spo:
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                        start_t_span = start_t - span_start + candidate_start
                        end_t_span = end_t - span_start + candidate_start
                        tmp_tuple_list.append((start_t_span, end_t_span-1))
                        # print(r, a, self.tokenizer.decode(input_ids[start_t:end_t]), self.tokenizer.decode(span_input_ids[start_t_span:end_t_span]))
                else:
                    a_token = self.tokenizer.encode(a, add_special_tokens=False)
                    a_list = search_all(a_token, input_ids[span_start:span_end])
                    if a_list:
                        for a_pos in a_list:
                            start_t_span = a_pos + candidate_start
                            end_t_span = a_pos + len(a_token) + candidate_start
                            tmp_tuple_list.append((start_t_span, end_t_span-1))
        
        entity_labels = [set() for i in range(1)]
        
        for sh, st in tmp_tuple_list:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
        for label in entity_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])

        return text, entity_labels, span_input_ids, span_attention_mask, span_type_ids 
                
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
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels