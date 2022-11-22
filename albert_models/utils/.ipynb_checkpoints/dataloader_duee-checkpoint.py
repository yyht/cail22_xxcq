import json
import numpy as np
import torch
from torch.utils.data import Dataset
from operator import is_not
from functools import partial
import random

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

def load_ee_as_ie(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            spo_list = []
            for e in line["event_list"]:
                if e.get('trigger', None):
                    for a in e['arguments']:
                        spo_list.append((e['trigger'], e['event_type'], a['argument'], 'trigger', a['role']))
            D.append({
                "text":line["text"],
                "spo_list":spo_list
            })
        return D

def load_name(filename):
    #{"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            d = {'text': line['text'], 'events': []}
            for e in line["event_list"]:
                if e.get('trigger', None):
                    d['events'].append([(
                        e['event_type'], u'触发词', e['trigger'],
                        e.get('trigger_start_index', -1)
                    )])
                else:
                    d['events'].append([])
                for a in e['arguments']:
                    d['events'][-1].append((
                        e['event_type'], a['role'], a['argument'],
                        a.get('argument_start_index', -1)
                    ))
            D.append(d)
        print(D[0])
        random.shuffle(D)
        return D
    
import re
def load_name_split(filename):
    #{"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            d = {'text': line['text'], 'events': []}
            for e in line["event_list"]:
                if e.get('trigger', None):
                    d['events'].append([(
                        e['event_type'], u'触发词', e['trigger'],
                        e.get('trigger_start_index', -1)
                    )])
                else:
                    d['events'] = [[]]
                for a in e['arguments']:
                    d['events'][-1].append((
                        e['event_type'], a['role'], a['argument'],
                        a.get('argument_start_index', -1)
                    ))
            for text in re.split('\n', line["text"]):
                if len(text) >= 2:
                    D.append({
                        "text":text,
                        "events":d['events']
                    })
            D.append(d)
        print(D[0])
        random.shuffle(D)
        return D

def load_data(filename, filter_event=False, schema_type_dict={}, ng_keep_rate=0.3, max_len=512):
    #{"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    D = []
    # ng_keep_rate = 0.3
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            d = {'text': line['text'], 'events': []}
            for e in line["event_list"]:
                
                # 在此处进行event_type筛选
                if filter_event:
                    if e['event_type'] not in schema_type_dict["体育竞赛"] \
                        and e['event_type'] not in schema_type_dict["灾害意外"]:
                        continue
                
                if e.get('trigger', None):
                    d['events'].append([(
                        e['event_type'], u'触发词', e['trigger'],
                        e.get('trigger_start_index', -1)
                    )])
                else:
                    d['events'] = [[]]
                
                for a in e['arguments']:
                    d['events'][-1].append((
                        e['event_type'], a['role'], a['argument'],
                        a.get('argument_start_index', -1)
                    ))
            # 此处可以增加负采样，是否需要保留events为空的信息
            if len(d['events']) == 0:
                continue
            else:  
                D.append(d)
            
        print(D[0])
        random.shuffle(D)
        return D
    
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

def search_all(pattern, sequence):
    all_index = []
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            all_index.append(i)
    return all_index

class data_generator(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, greedy_search=False, seg_token='<S>'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo (event_type, role_type)
        self.labels = [label] * len(self.data)
        self.greedy_search = greedy_search
        
        self.seg_token = seg_token
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token] })
        
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        
        token2char = encoder_text.offset_mapping
        char2token = [None] * len(text)
        for i, ((start, end)) in enumerate(token2char):
            char2token[start:end] = [i] * (end - start)
        
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        events = []
        for e in item['events']:
            events.append([])
            for e, r, a, i in e:
                label = self.schema[(e, r)]
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t]:
                        events[-1].append((label, start_t, end_t-1))
                    if self.greedy_search:
                        argument_ids = self.tokenizer.encode(a, add_special_tokens=False)
                        a_list = search_all(argument_ids, input_ids)
                        if a_list:
                            for a_i in a_list:
                                start, end = a_i, a_i + len(argument_ids) - 1
                                events[-1].append((label, start, end))
                else:
                    if self.greedy_search:
                        argument_ids = self.tokenizer.encode(a, add_special_tokens=False)
                        a_list = search_all(argument_ids, input_ids)
                        if a_list:
                            for a_i in a_list:
                                start, end = a_i, a_i + len(argument_ids) - 1
                                events[-1].append((label, start, end))
        # 构建标签
        # print(events, '===')
        argu_labels = [set() for _ in range(len(self.schema))]
        head_labels, tail_labels = set(), set()
        for e in events:
            for l, h, t in e:
                argu_labels[l].add((h, t))
            for i1, (_, h1, t1) in enumerate(e):
                for i2, (_, h2, t2) in enumerate(e):
                    if i2 > i1:
                        head_labels.add((min(h1, h2), max(h1, h2)))
                        tail_labels.add((min(t1, t2), max(t1, t2)))
        for label in argu_labels + [head_labels, tail_labels]:
            if not label:  # 至少要有一个标签
                label.add((0, 0))  # 如果没有则用0填充

        argu_labels = sequence_padding([list(l) for l in argu_labels])
        head_labels = sequence_padding([list(head_labels)])
        tail_labels = sequence_padding([list(tail_labels)])

        return text, argu_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_argu_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, argu_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_argu_labels.append(argu_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_argu_labels = torch.tensor(sequence_padding(batch_argu_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_argu_labels, batch_head_labels, batch_tail_labels
    
class data_generator_qa(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, template=None, label=0, greedy_search=False, seg_token='<S>'):
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo (event_type, role_type)
        
        self.schema_role_type_dict = {}
        for (et, rt) in self.schema:
            if et not in self.schema_role_type_dict:
                self.schema_role_type_dict[et] = set()
            self.schema_role_type_dict[et].add(rt)
            
        print(self.schema_role_type_dict, '==schema_role_type_dict==')
        
        self.greedy_search = greedy_search
        
        self.seg_token = seg_token
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token] })
        self.template = template
        self.data = []
        
        import random
        
        for item in data:
            events = {}
            for event in item['events']:
                for (e, r, a, i) in event:
                    if e not in events:
                        events[e] = {}
                    if r not in events[e]:
                        events[e][r] = set()
                    events[e][r].add((e, r, a, i))
            
            for e in events:
                for r in events[e]:
                    tmp = {
                        'text':item['text'],
                        'events':list(events[e][r]) # only for positive extraction
                    }
                    self.data.append(tmp)
                
                neg_r = list(set(self.schema_role_type_dict[e]) - set(list(events[e].keys())))
                random.shuffle(neg_r)
                
                for r in neg_r:
                    tmp = {
                        'text':item['text'],
                        'events':[(e, r, '', -1)]
                    }
                    self.data.append(tmp)
                
                # if len(events[e]) == 1 and '触发词' in list(events[e].keys()):
                #     # print(neg_r, '==neg==', self.schema_role_type_dict[e], '==gold==', list(events[e].keys()), '==used==')
                #     if len(neg_r) >= 1:
                #         for r in neg_r:
                #             tmp = {
                #                 'text':item['text'],
                #                 'events':[(e, r, '', -1)]
                #             }
                #             self.data.append(tmp)
                # else:
                #     if len(neg_r) >= 1:
                #         for r in neg_r[:2]:
                #             tmp = {
                #                 'text':item['text'],
                #                 'events':[(e, r, '', -1)]
                #             }
                #             self.data.append(tmp)
                    
        self.labels = [label] * len(self.data)
        
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        
        token2char = encoder_text.offset_mapping
        char2token = [None] * len(text)
        for i, ((start, end)) in enumerate(token2char):
            char2token[start:end] = [i] * (end - start)
        
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        events = set()
        
        for e, r, a, i in item['events']:
            if i != -1 and a != '':
                start, end = i, i + len(a) - 1
                start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                if input_ids[start_t:end_t]:
                    events.add((start_t, end_t-1))

        # 构建标签
        entity_labels = [set() for i in range(1)]
        
        for (h, t) in events:
            entity_labels[0].add((h, t))
        for label in entity_labels:
            if not label:  # 至少要有一个标签
                label.add((0, 0))  # 如果没有则用0填充

        entity_labels = sequence_padding([list(l) for l in entity_labels])
        
        et_text = self.tokenizer('{}'.format(e), return_offsets_mapping=True, max_length=32)
        input_ids += et_text['input_ids'][1:]
        token_type_ids +=  et_text['token_type_ids'][1:]
        attention_mask += et_text['attention_mask'][1:]
        
        at_text = self.tokenizer('{}'.format(r), return_offsets_mapping=True, max_length=32)
        input_ids += at_text['input_ids'][1:]
        token_type_ids +=  at_text['token_type_ids'][1:]
        attention_mask += at_text['attention_mask'][1:]
                        
        return text, entity_labels, input_ids, attention_mask, token_type_ids, events

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels = []
        batch_entity_labels_dense = []
        batch_start_end_labels = []
        batch_loss_mask = []
        text_list = []
        event_list = []
        for item in examples:
            text, entity_labels, input_ids, attention_mask, token_type_ids, events = item
            batch_entity_labels.append(entity_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)
            event_list.append(events)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        
        max_len = batch_token_ids.shape[1]
        for events in event_list:
            entity_labels_dense = np.zeros((max_len, max_len))
            start_end_labels = np.zeros((max_len, 2))
            for (h,t) in events:
                entity_labels_dense[h, t] = 1.0
                start_end_labels[h, 0] = 1.0
                start_end_labels[t, 1] = 1.0
            batch_start_end_labels.append(start_end_labels)
            batch_entity_labels_dense.append(entity_labels_dense)
        batch_entity_labels_dense = torch.tensor(batch_entity_labels_dense).float()
        batch_start_end_labels = torch.tensor(batch_start_end_labels).float()
        
        return [text_list, batch_token_ids, batch_mask_ids, 
                batch_token_type_ids, batch_entity_labels, batch_entity_labels_dense,
               batch_start_end_labels]
    
class data_generator_cls_then_detection(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, schema2universal, 
                 unisersal_schema, 
                 et_schema, label=0, mode='cls', seg_token='[SEP]', apply_universal_schema=False,
                add_neg=False, greedy_search=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo (event_type, role_type)
        self.schema2universal = schema2universal # (event_type, role_type)--> standard schema only related to role_type
        self.unisersal_schema = unisersal_schema
        self.mode = mode
        self.seg_token = seg_token
        self.apply_universal_schema = apply_universal_schema
        self.et_schema = et_schema
        self.add_neg = add_neg
        self.greedy_search = greedy_search
        
        self.id2unisersal_schema = {}
        for s in self.unisersal_schema:
            idx = self.unisersal_schema[s]
            self.id2unisersal_schema[idx] = s
        
        self.features = []
        break_flag = False
        for item in self.data:
            for feature in self.encoder(item):
                self.features.append(feature)
        
        self.labels = [label] * len(self.features)
        
    def __len__(self):
        return len(self.features)

    def encoder(self, item):
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        
        token2char = encoder_text.offset_mapping
        char2token = [None] * len(text)
        for i, ((start, end)) in enumerate(token2char):
            char2token[start:end] = [i] * (end - start)
        
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        et_events = {}
        
        for e in item['events']:
            for index, (e, r, a, i) in enumerate(e):
                if index == 0:
                    if e not in et_events:
                        et_events[e] = []
                    et_events[e].append([])
                label = (e, r)
                if self.apply_universal_schema:
                    label = self.schema2universal[label] # universal_shcmea
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t]:
                        et_events[e][-1].append((label, start_t, end_t-1))
                        # print(label, self.tokenizer.decode(input_ids[start_t:end_t]), self.unisersal_schema[label], '======')
                else:
                    if self.greedy_search:
                        argument_ids = self.tokenizer.encode(a, add_special_tokens=False)
                        a_list = search_all(argument_ids, input_ids)
                        if a_list:
                            for a_i in a_list:
                                start, end = a_i, a_i + len(argument_ids) - 1
                                et_events[e][-1].append((label, start, end))
        
        output_tuple_list = []
        if self.mode == 'cls':
            et_labels = [0] * len(self.et_schema)
            for e_t in et_events:
                et_labels[self.et_schema[e_t]] = 1
            output_tuple_list.append((text, et_labels, input_ids, attention_mask, token_type_ids))
        
        elif self.mode == 'p_extraction':
            pos_et = set()
            for e_t in et_events:
                
                et_ids = self.tokenizer(e_t)['input_ids'][1:] #  [cls] text [sep] relation [sep]
                new_input_ids = input_ids + et_ids    
                token_type_ids = [0]*len(new_input_ids)
                attention_mask = [1]*len(new_input_ids)
                
                pos_et.add(e_t)
                # 构建标签
                
                if self.apply_universal_schema:
                    argu_labels = [set() for _ in range(len(self.unisersal_schema))]
                else:
                    argu_labels = [set() for _ in range(len(self.schema))]
                
                head_labels, tail_labels = set(), set()
                for e in et_events[e_t]:
                    for l, h, t in e:
                        if self.apply_universal_schema:
                            argu_labels[self.unisersal_schema[l]].add((h, t))
                        else:
                            argu_labels[self.schema[l]].add((h, t))
                        
                    for i1, (_, h1, t1) in enumerate(e):
                        for i2, (_, h2, t2) in enumerate(e):
                            if i2 > i1:
                                head_labels.add((min(h1, h2), max(h1, h2)))
                                tail_labels.add((min(t1, t2), max(t1, t2)))
                
                # for index, argu in enumerate(argu_labels):
                #     if argu:
                #         for (h, t) in argu:
                #             print(self.tokenizer.decode(new_input_ids[h:t+1]), self.id2unisersal_schema[index], self.tokenizer.decode(new_input_ids))
                
                for label in argu_labels + [head_labels, tail_labels]:
                    if not label:  # 至少要有一个标签
                        label.add((0, 0))  # 如果没有则用0填充
                        
                argu_labels = sequence_padding([list(l) for l in argu_labels])
                head_labels = sequence_padding([list(head_labels)])
                tail_labels = sequence_padding([list(tail_labels)])
                
                output_tuple_list.append(((text, e_t), argu_labels, head_labels, tail_labels, new_input_ids, attention_mask, token_type_ids))
            
            if self.add_neg:
                all_et = set(list(self.et_schema.keys()))
                neg_et = list(all_et - pos_et)
                random.shuffle(neg_et)
                for e_t in neg_et[:2]:
                    if self.apply_universal_schema:
                        argu_labels = [set() for _ in range(len(self.unisersal_schema))]
                    else:
                        argu_labels = [set() for _ in range(len(self.schema))]
                    head_labels, tail_labels = set(), set()
                    for label in argu_labels + [head_labels, tail_labels]:
                        if not label:  # 至少要有一个标签
                            label.add((0, 0))  # 如果没有则用0填充
                    argu_labels = sequence_padding([list(l) for l in argu_labels])
                    head_labels = sequence_padding([list(head_labels)])
                    tail_labels = sequence_padding([list(tail_labels)])

                    et_ids = self.tokenizer(e_t)['input_ids'][1:] #  [cls] text [sep] event-type [sep]
                    new_input_ids = input_ids + et_ids    
                    token_type_ids = [0]*len(new_input_ids)
                    attention_mask = [1]*len(new_input_ids)
                    output_tuple_list.append(((text, e_t), rgu_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids))
                    
        return output_tuple_list

    def __getitem__(self, idx):
        return self.features[idx]
    
    def get_labels(self):
        return self.labels
    
    @staticmethod
    def collate_cls(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_labels = []
        text_list = []
        for item in examples:
            text, labels, input_ids, attention_mask, token_type_ids = item
            batch_labels.append(labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_labels = torch.tensor(batch_labels).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_labels

    @staticmethod
    def collate_p_extraction(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_argu_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, argu_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_argu_labels.append(argu_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_argu_labels = torch.tensor(sequence_padding(batch_argu_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_argu_labels, batch_head_labels, batch_tail_labels
    
    
class data_generator_nce(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, predicate_schema, 
                 template=None, label=0, greedy_search=False, seg_token='<S>', add_neg=False):
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema # role_type
        self.predicate_schema = predicate_schema
                
        self.greedy_search = greedy_search
        
        self.seg_token = seg_token
        self.add_neg = add_neg
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token] })
        self.template = template
        self.data = []
        
        import random
        
        neg_cnt = 0
        
        for item in data:
            et_events = {}
            for event in item['events']:
                for index, (e, r, a, i) in enumerate(event):
                    if index == 0:
                        if e not in et_events:
                            et_events[e] = []
                        et_events[e].append([])
                    et_events[e][-1].append((e, r, a, i))
            
            for et in et_events:
                tmp = {
                    'text':item['text'],
                    'events':et_events[et], # only for positive extraction,
                    'event_type': et
                }
                self.data.append(tmp)
                
            neg_e = list(set(list(self.predicate_schema.keys())) - set(list(et_events.keys())))
            random.shuffle(neg_e)
            if len(neg_e) >= 1 and self.add_neg:
                neg_cnt += 1
                tmp = {
                        'text':item['text'],
                        'events':[],
                        'candidate_type': neg_e

                }
                self.data.append(tmp)
                
        self.labels = [label] * len(self.data)
        print('====size of dataset====', len(self.data), '--neg count==', neg_cnt)
        
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        
        if 'candidate_type' in item:
            import random
            random.shuffle(item['candidate_type'])
            et = item['candidate_type'][0]
        else:
            et = item['event_type']
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        
        token2char = encoder_text.offset_mapping
        char2token = [None] * len(text)
        for i, ((start, end)) in enumerate(token2char):
            char2token[start:end] = [i] * (end - start)
        
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]

        events = []
        for e in item['events']:
            events.append([])
            for e, r, a, i in e:
                label = self.schema[r]
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t]:
                        events[-1].append((label, start_t, end_t-1))
                else:
                    if self.greedy_search:
                        argument_ids = self.tokenizer.encode(a, add_special_tokens=False)
                        a_list = search_all(argument_ids, input_ids)
                        if a_list:
                            for a_i in a_list:
                                start, end = a_i, a_i + len(argument_ids) - 1
                                events[-1].append((label, start, end))                    
        
        # 构建标签
        # print(events, '====', item)
        argu_labels = [set() for _ in range(len(self.schema))]
        head_labels, tail_labels = set(), set()
        for e in events:
            for l, h, t in e:
                argu_labels[l].add((h, t))
            for i1, (_, h1, t1) in enumerate(e):
                for i2, (_, h2, t2) in enumerate(e):
                    if i2 > i1:
                        head_labels.add((min(h1, h2), max(h1, h2)))
                        tail_labels.add((min(t1, t2), max(t1, t2)))
        for label in argu_labels + [head_labels, tail_labels]:
            if not label:  # 至少要有一个标签
                label.add((0, 0))  # 如果没有则用0填充

        argu_labels = sequence_padding([list(l) for l in argu_labels])
        head_labels = sequence_padding([list(head_labels)])
        tail_labels = sequence_padding([list(tail_labels)])
        
        et_text = self.tokenizer('{}'.format(et), return_offsets_mapping=True, max_length=64)
        input_ids += et_text['input_ids'][1:]
        token_type_ids +=  et_text['token_type_ids'][1:]
        attention_mask += et_text['attention_mask'][1:]

        return text, argu_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_argu_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, argu_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_argu_labels.append(argu_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_argu_labels = torch.tensor(sequence_padding(batch_argu_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_argu_labels, batch_head_labels, batch_tail_labels
    