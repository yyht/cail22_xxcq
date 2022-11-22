import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import itertools
from operator import is_not
from functools import partial
import re
        
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

def mat_padding(inputs, length=None, padding=0):

    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[0] for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        pad_width[0] = (0, length - x.shape[0])
        pad_width[1] = (0, length - x.shape[0])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

class data_generator_negative_learning_negative_balanced(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, doc_stride=32, offset=8,
                 seg_token='<S>', sep_token='[SEP]', start_token='[CLS]', link_symbol='-',
                 mode='upsampling'):
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
        self.label_list = ["N/A","SMH","SMT","SS","MMH","MMT","MSH","MST"]
        self.id2label, self.label2id={},{}
        for i,l in enumerate(self.label_list):
            self.id2label[str(i)]=l
            self.label2id[l]=i
        
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

        spoes = {}
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
                if (tmp_tuple[0], tmp_tuple[1]) not in spoes:
                    spoes[(tmp_tuple[0], tmp_tuple[1])] = [(tmp_tuple[2], tmp_tuple[3])]
                spoes[(tmp_tuple[0], tmp_tuple[1])].append((tmp_tuple[2], tmp_tuple[3]))
            elif len(tmp_tuple_list) == 2:
                tuple_list = beam_search(tmp_tuple_list)
                for item_pair in tuple_list:
                    tmp_tuple_ = item_pair[0]
                    tmp_tuple_ += item_pair[1]
                    if len(tmp_tuple_) == 4:
                        if (tmp_tuple_[0], tmp_tuple_[1]) not in spoes:
                            spoes[(tmp_tuple_[0], tmp_tuple_[1])] = []
                        spoes[(tmp_tuple_[0], tmp_tuple_[1])].append((tmp_tuple_[2], tmp_tuple_[3]))
        # if not spoes:
        #     print(item['spoes'], text)
        
        label=np.zeros([len(span_input_ids), len(span_input_ids),1]) #LLR
        mask_label=np.ones(label.shape)
        #label = ["N/A", "SMH", "SMT", "SS", "MMH", "MMT", "MSH","MST"]
        for s in spoes:
            s1,s2=s
            for o1,o2 in spoes[s]:
                if s1==s2 and o1==o2:
                    label[s1,o1,0]=self.label2id["SS"]
                elif s1!=s2 and o1==o2:
                    label[s1,o1,0]=self.label2id["MSH"]
                    label[s2,o1,0]=self.label2id["MST"]
                elif s1==s2 and o1!=o2:
                    label[s1,o1,0]=self.label2id["SMH"]
                    label[s1,o2,0]=self.label2id["SMT"]
                elif s1!=s2 and o1!=o2:
                    label[s1, o1,0] = self.label2id["MMH"]
                    label[s2, o2,0] = self.label2id["MMT"]

        mask_label[0:candidate_start,:,:]=0
        mask_label[-1,:,:]=0
        mask_label[:,0:candidate_start,:]=0
        mask_label[:,-1,:]=0
        return text, label, mask_label, span_input_ids, span_attention_mask, span_type_ids 
           
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
        batch_labels, batch_mask_labels = [], []
        text_list = []
        for item in examples:
            text, labels, mask_labels, input_ids, attention_mask, token_type_ids = item
            batch_labels.append(labels)
            batch_mask_labels.append(mask_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_labels = torch.tensor(mat_padding(batch_labels)).long()
        batch_mask_labels = torch.tensor(mat_padding(batch_mask_labels)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_labels, batch_mask_labels