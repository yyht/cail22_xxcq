import torch
from torch import nn
import json
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


class UniMCDataset(Dataset):
    def __init__(self, data, yes_token, no_token, tokenizer, args, used_mask=True):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.num_labels = args.num_labels
        self.used_mask = used_mask
        self.data = data
        self.args = args
        self.yes_token = yes_token
        self.no_token = no_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data[index], self.used_mask)

    def get_token_type(self, sep_idx, max_length):
        token_type_ids = np.zeros(shape=(max_length,))
        for i in range(len(sep_idx)-1):
            if i % 2 == 0:
                ty = np.ones(shape=(sep_idx[i+1]-sep_idx[i],))
            else:
                ty = np.zeros(shape=(sep_idx[i+1]-sep_idx[i],))
            token_type_ids[sep_idx[i]:sep_idx[i+1]] = ty

        return token_type_ids

    def get_position_ids(self, label_idx, max_length, question_len):
        question_position_ids = np.arange(question_len)
        label_position_ids = np.arange(question_len, label_idx[-1])
        for i in range(len(label_idx)-1):
            label_position_ids[label_idx[i]-question_len:label_idx[i+1]-question_len] = np.arange(
                question_len, question_len+label_idx[i+1]-label_idx[i])
        max_len_label = max(label_position_ids)
        text_position_ids = np.arange(
            max_len_label+1, max_length+max_len_label+1-label_idx[-1])
        position_ids = list(question_position_ids) + \
            list(label_position_ids)+list(text_position_ids)
        if max_length <= 512:
            return position_ids[:max_length]
        else:
            for i in range(512, max_length):
                if position_ids[i] > 511:
                    position_ids[i] = 511
            return position_ids[:max_length]

    def get_att_mask(self, attention_mask, label_idx, question_len):
        max_length = len(attention_mask)
        attention_mask = np.array(attention_mask)
        attention_mask = np.tile(attention_mask[None, :], (max_length, 1))

        zeros = np.zeros(
            shape=(label_idx[-1]-question_len, label_idx[-1]-question_len))
        attention_mask[question_len:label_idx[-1],
                       question_len:label_idx[-1]] = zeros

        for i in range(len(label_idx)-1):
            label_token_length = label_idx[i+1]-label_idx[i]
            if label_token_length <= 0:
                print('label_idx', label_idx)
                print('question_len', question_len)
                continue
            ones = np.ones(shape=(label_token_length, label_token_length))
            attention_mask[label_idx[i]:label_idx[i+1],
                           label_idx[i]:label_idx[i+1]] = ones

        return attention_mask

    def random_masking(self, token_ids, maks_rate, mask_start_idx, max_length, mask_id, tokenizer):
        rands = np.random.random(len(token_ids))
        source, target = [], []
        for i, (r, t) in enumerate(zip(rands, token_ids)):
            if i < mask_start_idx:
                source.append(t)
                target.append(-100)
                continue
            if r < maks_rate * 0.8:
                source.append(mask_id)
                target.append(t)
            elif r < maks_rate * 0.9:
                source.append(t)
                target.append(t)
            elif r < maks_rate:
                source.append(np.random.choice(tokenizer.vocab_size - 1) + 1)
                target.append(t)
            else:
                source.append(t)
                target.append(-100)
        while len(source) < max_length:
            source.append(0)
            target.append(-100)
        return source[:max_length], target[:max_length]

    def encode(self, item, used_mask=False):

        while len(self.tokenizer.encode('[MASK]'.join(item['choice']))) > self.max_length-32:
            item['choice'] = [c[:int(len(c)/2)] for c in item['choice']]

        if 'textb' in item.keys() and item['textb'] != '':
            if 'question' in item.keys() and item['question'] != '':
                texta = '[MASK]' + '[MASK]'.join(item['choice']) + '[SEP]' + \
                    item['question'] + '[SEP]' + \
                        item['texta']+'[SEP]'+item['textb']
            else:
                texta = '[MASK]' + '[MASK]'.join(item['choice']) + '[SEP]' + \
                        item['texta']+'[SEP]'+item['textb']

        else:
            if 'question' in item.keys() and item['question'] != '':
                texta = '[MASK]' + '[MASK]'.join(item['choice']) + '[SEP]' + \
                    item['question'] + '[SEP]' + item['texta']
            else:
                texta = '[MASK]' + '[MASK]'.join(item['choice']) + \
                    '[SEP]' + item['texta']

        encode_dict = self.tokenizer.encode_plus(texta,
                                                 max_length=self.max_length,
                                                 padding='max_length',
                                                 truncation='longest_first')

        encode_sent = encode_dict['input_ids']
        token_type_ids = encode_dict['token_type_ids']
        attention_mask = encode_dict['attention_mask']
        sample_max_length = sum(encode_dict['attention_mask'])

        if 'label' not in item.keys():
            item['label'] = 0
            item['answer'] = ''

        question_len = 1
        label_idx = [question_len]
        for choice in item['choice']:
            cur_mask_idx = label_idx[-1] + \
                len(self.tokenizer.encode(choice, add_special_tokens=False))+1
            label_idx.append(cur_mask_idx)

        token_type_ids = [0]*question_len+[1] * \
            (label_idx[-1]-label_idx[0]+1)+[0]*self.max_length
        token_type_ids = token_type_ids[:self.max_length]

        attention_mask = self.get_att_mask(
            attention_mask, label_idx, question_len)

        position_ids = self.get_position_ids(
            label_idx, self.max_length, question_len)

        clslabels_mask = np.zeros(shape=(len(encode_sent),))
        clslabels_mask[label_idx[:-1]] = 10000
        clslabels_mask = clslabels_mask-10000

        mlmlabels_mask = np.zeros(shape=(len(encode_sent),))
        mlmlabels_mask[label_idx[0]] = 1

        # used_mask=False
        if used_mask:
            mask_rate = 0.1*np.random.choice(4, p=[0.3, 0.3, 0.25, 0.15])
            source, target = self.random_masking(token_ids=encode_sent, maks_rate=mask_rate,
                                                 mask_start_idx=label_idx[-1], max_length=self.max_length,
                                                 mask_id=self.tokenizer.mask_token_id, tokenizer=self.tokenizer)
        else:
            source, target = encode_sent[:], encode_sent[:]

        source = np.array(source)
        target = np.array(target)
        source[label_idx[:-1]] = self.tokenizer.mask_token_id
        target[label_idx[:-1]] = self.no_token
        target[label_idx[item['label']]] = self.yes_token

        input_ids = source[:sample_max_length]
        token_type_ids = token_type_ids[:sample_max_length]
        attention_mask = attention_mask[:sample_max_length, :sample_max_length]
        position_ids = position_ids[:sample_max_length]
        mlmlabels = target[:sample_max_length]
        clslabels = label_idx[item['label']]
        clslabels_mask = clslabels_mask[:sample_max_length]
        mlmlabels_mask = mlmlabels_mask[:sample_max_length]

        return {
            "input_ids": torch.tensor(input_ids).long(),
            "token_type_ids": torch.tensor(token_type_ids).long(),
            "attention_mask": torch.tensor(attention_mask).float(),
            "position_ids": torch.tensor(position_ids).long(),
            "mlmlabels": torch.tensor(mlmlabels).long(),
            "clslabels": torch.tensor(clslabels).long(),
            "clslabels_mask": torch.tensor(clslabels_mask).float(),
            "mlmlabels_mask": torch.tensor(mlmlabels_mask).float(),
        }
    
    @staticmethod
    def collate_fn(batch):
        '''
        Aggregate a batch data.
        batch = [ins1_dict, ins2_dict, ..., insN_dict]
        batch_data = {'sentence':[ins1_sentence, ins2_sentence...], 'input_ids':[ins1_input_ids, ins2_input_ids...], ...}
        '''
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]

        batch_data['input_ids'] = nn.utils.rnn.pad_sequence(batch_data['input_ids'],
                                                            batch_first=True,
                                                            padding_value=0)
        batch_data['clslabels_mask'] = nn.utils.rnn.pad_sequence(batch_data['clslabels_mask'],
                                                                 batch_first=True,
                                                                 padding_value=-10000)

        batch_size, batch_max_length = batch_data['input_ids'].shape
        for k, v in batch_data.items():
            if k == 'input_ids' or k == 'clslabels_mask':
                continue
            if k == 'clslabels':
                batch_data[k] = torch.tensor(v).long()
                continue
            if k != 'attention_mask':
                batch_data[k] = nn.utils.rnn.pad_sequence(v,
                                                          batch_first=True,
                                                          padding_value=0)
            else:
                attention_mask = torch.zeros(
                    (batch_size, batch_max_length, batch_max_length))
                for i, att in enumerate(v):
                    sample_length, _ = att.shape
                    attention_mask[i, :sample_length, :sample_length] = att
                batch_data[k] = attention_mask
        return batch_data