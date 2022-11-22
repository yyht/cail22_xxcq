import json
import numpy as np
import torch
from torch.utils.data import Dataset
from operator import is_not
from functools import partial

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

def load_name(filename):
    #{"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            d = {'text': text, 'events': []}
            for e in l.get('event_list', []):
                d['events'].append([(
                    e['event_type'], u'触发词', (e['trigger']),
                    text.index(e['trigger'])
                )])
                for a in e['arguments']:
                    a['argument'] = normalize(a['argument'])
                    if (e['event_type'], a['role']) in enum_roles:
                        d['events'][-1].append((
                            e['event_type'], a['argument'], e['trigger'],
                            text.index(e['trigger'])
                        ))
                    else:
                        d['events'][-1].append((
                            e['event_type'], a['role'], a['argument'],
                            text.index(a['argument'])
                        ))
            D.append(d)
        print(D[0])
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

class data_generator(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo (event_type, role_type)
        self.labels = [label] * len(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def encoder(self, item):
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)  
        
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
                start, end = i, i + len(a) - 1
                start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                if input_ids[start_t:end_t]:
                    events[-1].append((label, start_t, end_t-1))
                
        # 构建标签
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