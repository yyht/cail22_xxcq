# -*- coding: utf-8 -*-
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from operator import is_not
from functools import partial
import random
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from collections import OrderedDict

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.augment import insert_punctuation_marks
import random
import logging
# from utils.flashtext import KeywordProcessor
from utils.keyword_processor import KeywordProcesser
from utils.utils import token_span_to_char_span, char_span_to_token_span

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from utils.mlm_generator import MLMGenerator
from utils.mask_utils import create_sentinel_ids, filter_input_ids, random_spans_noise_mask
import jieba_fast

class AutoTemplate(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False,
                doc_stride=32, offset=0, stop_words={}, keywords={}):

        self.data = data
        self.doc_stride = doc_stride
        self.offset = offset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_dict = task_dict
        self.sep_token = task_dict['sep_token']
        self.seg_token = task_dict['seg_token']
        self.group_token = task_dict['group_token']
        self.start_token = task_dict['start_token']
        self.end_token = task_dict['end_token']
        self.sentinel_token = task_dict['sentinel_token']
        self.sentinel_start_idx = task_dict['sentinel_start_idx']
        self.stop_words = stop_words
        self.keywords = keywords
        
        self.stopword_api = KeywordProcesser()
        self.keyword_api = KeywordProcesser()
        self.special_keyword_api = KeywordProcesser()
        for stop_word in self.stop_words:
            self.stopword_api.add_keyword(stop_word)
        for keyword in self.keywords:
            self.keyword_api.add_keyword(keyword, {'type':'key_word', 'info':self.keywords[keyword]})

        self.add_spefical_tokens = self.task_dict.get('add_spefical_tokens', True)
        if self.add_spefical_tokens:
            self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [ self.sentinel_token.format(i) for i in range(1, 100)]+[self.seg_token, self.sep_token, 
                                                                                                self.group_token, self.start_token, self.end_token] })
        self.mlm_generator = MLMGenerator(
                                self.task_dict.get('mask_ratio', 0.25), 
                                self.task_dict.get('random_ratio', 1e-10),
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
        
        import random
        random.shuffle(self.features)
        self.labels = [label] * len(self.features)
        self._task_id = label
                                                                                                          
    def prepare_ilm(self, span_input_ids, masked_target, tokenizer, **kargs):
        mask_indices = np.array([masked_target != 0]) # [ 0  1 -1  0  0  0  0  2 -1 -1  0  0  0  0  3 -1 -1 -1]
        input_ids_sentinel = create_sentinel_ids(mask_indices.astype(np.int8), tokenizer, 'unilm')[0]
        ilm_input_ids = []
        ilm_target_ids = []
        
        if_add_sentinel = True
        for idx, mask_indice in enumerate(input_ids_sentinel):
            if mask_indice == 0:
                ilm_input_ids += [span_input_ids[idx]]
                if not if_add_sentinel:
                    ilm_target_ids += encoder_sentinel['input_ids']
                    if_add_sentinel = True
            if mask_indice > 0:
                if not if_add_sentinel:
                    ilm_target_ids += encoder_sentinel['input_ids']
                    if_add_sentinel = True
                
                encoder_sentinel = self.tokenizer(self.sentinel_token.format(self.sentinel_start_idx+mask_indice), return_offsets_mapping=True, truncation=False, add_special_tokens=False)
                ilm_input_ids += encoder_sentinel['input_ids']
                ilm_target_ids += [span_input_ids[idx]]
                if_add_sentinel = False

            if mask_indice < 0:
                ilm_target_ids += [span_input_ids[idx]]
                if_add_sentinel = False

        # if mask_indice < 0:
        if not if_add_sentinel:
            ilm_target_ids += encoder_sentinel['input_ids']
                                                                                                          
        return ilm_input_ids, ilm_target_ids
                                                                                                        
    def prepare_t5(self, span_input_ids, masked_target, tokenizer, **kargs):
        mask_indices = np.array([masked_target != 0]) # [ 0  1 -1  0  0  0  0  2 -1 -1  0  0  0  0  3 -1 -1 -1]
        labels_mask = ~mask_indices

        input_ids_sentinel = create_sentinel_ids(mask_indices.astype(np.int8), tokenizer, 't5')
        # print(input_ids_sentinel, '==input_ids_sentinel==')

        labels_sentinel = create_sentinel_ids(labels_mask.astype(np.int8), tokenizer, 't5')
        # print(labels_sentinel, '==labels_sentinel==')

        ilm_input_ids = filter_input_ids(np.array([span_input_ids]), input_ids_sentinel)[0]
        ilm_target_ids = filter_input_ids(np.array([span_input_ids]), labels_sentinel)[0]
                                                                                                          
        return ilm_input_ids, ilm_target_ids

    def encoder(self, item):
                                                                                                          
        instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
        encoder_instruction_text = self.tokenizer(instruction_text, return_offsets_mapping=True, max_length=self.max_len, truncation=True, add_special_tokens=False)
        instruction_input_ids = encoder_instruction_text["input_ids"]
        instruction_start = len(instruction_input_ids)
        
        input_ids = item['input_ids']
        span_start = item['span_start']
        span_end = item['span_end']
        token2char = item['token2char']
        char2token = item['char2token']
        text = item['text']
        query = item.get('query', '') # extra attribute or content
        keywords = item.get('keywords', [])

        query += self.end_token
        encoder_query_text = self.tokenizer(query, return_offsets_mapping=True, max_length=self.max_len, truncation=True, add_special_tokens=False)

        span_input_ids = input_ids[span_start:span_end]
                                                                     
        # convert to original text position under current slide-window
        span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
        
        # convert to original text under current slide-window
        span_text = text[span_text_pos[0]:span_text_pos[1]]
        
        # extract keyword under current slide-window
        span_keywords = []
        # add provided keyword
        if keywords:
            for keyword in keywords:
                self.special_keyword_api.add_keyword(keyword)
            special_span_keywords = self.special_keyword_api.extract_keywords(span_text)
            if special_span_keywords:
                span_keywords.extend(special_span_keywords)
            for keyword in keywords:                                                                                 
                self.special_keyword_api.delete_keyword(keyword)
        
        # add general keyword
        general_span_keywords = self.keyword_api.extract_keywords(span_text)
        if general_span_keywords:
            span_keywords.extend(general_span_keywords)

        # add jieba cut
        words = list(jieba.cut(span_text))
        words = self.stopword_api.remove_keywords_in_words(words)
        for word in words:
            self.special_keyword_api.add_keyword(word)
        special_span_keywords = self.special_keyword_api.extract_keywords(span_text)
        if special_span_keywords:
            span_keywords.extend(special_span_keywords)

        ner_span = set()
        for span_word in span_keywords:
            # convert to original position in text
            start = span_word[0] + span_text_pos[0]
            end = span_word[1] + span_text_pos[0] - 1
            # convert to slide-window token position
            start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
            if start_t >= span_start and end_t <= span_end:
                span_start_t = start_t - span_start + instruction_start
                span_end_t = end_t - span_start + instruction_start
                ner_span.add((span_start_t, span_end_t-1))

        ner_span = list([list(value) for value in ner_span])
        
        random.shuffle(ner_span)
        lexicon_constraint_num = np.random.randint(1, min([10, len(ner_span)-1]))
        
        mask_num = len(span_input_ids) * 0.8
        
        [masked_sent, 
          masked_target, 
          _] = self.mlm_generator.template_span_mask(
                        span_input_ids, 
                        self.tokenizer,
                        entity_spans=ner_span[lexicon_constraint_num:],
                        return_only_spans=False,
                        ner_masking_prob=0.2,
                        mask_num=mask_num
                       )

        if self.task_dict.get('enc_dec_type', 'unilm') == 'unilm':
            ilm_input_ids, ilm_target_ids = self.prepare_ilm(span_input_ids, masked_target, self.tokenizer)                                                                                     
        elif self.task_dict.get('enc_dec_type', 'unilm') == 't5':
            ilm_input_ids, ilm_target_ids = self.prepare_t5(span_input_ids, masked_target, self.tokenizer)                                                                                           
        encoder_end_text = self.tokenizer(self.end_token, return_offsets_mapping=True, max_length=self.max_len, truncation=True, add_special_tokens=False)

        ilm_input_ids += encoder_query_text['input_ids']
        ilm_target_ids += encoder_end_text['input_ids']

        ilm_input_attention_mask = [1] * len(ilm_input_ids)
        ilm_input_token_type_ids = [0] * len(ilm_input_ids)
            
        ilm_target_attention_mask += [1] * len(ilm_target_ids)
        ilm_target_token_type_ids += [0] * len(ilm_target_ids)

        return (text, ilm_input_ids, ilm_input_attention_mask, ilm_input_token_type_ids,
            ilm_target_ids, ilm_target_attention_mask, ilm_target_token_type_ids)

    @staticmethod
    def collate_unilm(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        for item in examples:
            (text, ilm_input_ids, ilm_input_attention_mask, ilm_input_token_type_ids,
               ilm_target_ids, ilm_target_attention_mask, ilm_target_token_type_ids) = item
            
            # cls instruction sep input sep schema sep target end
            input_ids = ilm_input_ids
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            
            input_ids += ilm_target_ids
            attention_mask += [1] * len(ilm_target_ids)
            token_type_ids += [1] * len(ilm_target_ids)
            
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids
    
    @staticmethod
    def collate_t5(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, ilm_input_ids, ilm_input_attention_mask, ilm_input_token_type_ids,
               ilm_target_ids, ilm_target_attention_mask, ilm_target_token_type_ids) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = ilm_input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = ilm_target_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(encoder_input_ids)
            batch_encoder_mask_ids.append(encdoer_attention_mask)
            batch_encoder_token_type_ids.append(encoder_token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)