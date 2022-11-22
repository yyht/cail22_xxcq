
import math
import numpy as np
from utils import mask_utils
from collections import OrderedDict

"""
https://github.com/facebookresearch/SpanBERT/blob/master/pretraining/fairseq/data/masking.py

masked_lm_positions_a = [i+1 for i, e in enumerate(masked_target_a) if e != 0]
masked_lm_ids_a = [e for i, e in enumerate(masked_target_a) if e != 0]
masked_lm_weights = [1]*len(masked_lm_positions)
"""

class MLMGenerator(object):
    def __init__(self, 
        mask_ratio, 
        random_ratio,
        min_tok,
        max_tok,
        mask_id,
        pad,
        geometric_p,
        vocab,
        max_pair_targets,
        replacement_method='word_piece',
        endpoints='',
        **kargs):
        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.min_tok = min_tok
        self.max_tok = max_tok
        self.mask_id = mask_id
        self.pad = pad
        self.tokens = [index for index, word in enumerate(vocab)]
        self.geometric_p = geometric_p
        self.max_pair_targets = max_pair_targets

        self.p = geometric_p
        self.len_distrib = [self.p * (1-self.p)**(i - self.min_tok) for i in range(self.min_tok, self.max_tok + 1)] if self.p >= 0 else None
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]
        self.lens = list(range(self.min_tok, self.max_tok + 1))
        self.paragraph_info = mask_utils.ParagraphInfo(vocab)
        self.replacement_method = replacement_method
        """
        if endpoints is external: index_range = np.arange(a+1, b)
        else: index_range = np.arange(a, b+1)
        """
        self.endpoints = endpoints
        self.max_pair_targets = max_pair_targets

    def random_mask(self, input_text, 
                                    tokenizer, **kargs):
        if not isinstance(input_text, list):
            sentence = tokenizer.encode(input_text, add_special_tokens=False)
        else:
            sentence = input_text
        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * self.mask_ratio)
        mask = np.random.choice(sent_length, mask_num, replace=False)
        return mask_utils.bert_masking(sentence, mask, self.tokens, self.pad, self.mask_id)

    def mask_entity(self, sentence, mask_num, word_piece_map, spans, mask, entity_spans):
        if len(entity_spans) > 0:
            entity_index = np.random.choice(range(len(entity_spans)))
            entity_span = entity_spans[entity_index]
            spans.append([entity_span[0], entity_span[0]])
            for idx in range(entity_span[0], entity_span[1] + 1):
                if len(mask) >= mask_num:
                    break
                spans[-1][-1] = idx
                mask.add(idx)
            entity_spans.pop(entity_index)

    def mask_random_span(self, sentence, mask_num, word_piece_map, spans, mask, span_len, anchor):
        # find word start, end
        # this also apply ngram and whole-word-mask for english
        left1, right1 = self.paragraph_info.get_word_start(sentence, anchor, word_piece_map), self.paragraph_info.get_word_end(sentence, anchor, word_piece_map)
        spans.append([left1, left1])
        for i in range(left1, right1):
            if len(mask) >= mask_num:
                break
            mask.add(i)
            spans[-1][-1] = i
        num_words = 1
        right2 = right1
        while num_words < span_len and right2 < len(sentence) and len(mask) < mask_num:
            # complete current word
            left2 = right2
            right2 = self.paragraph_info.get_word_end(sentence, right2, word_piece_map)
            num_words += 1
            for i in range(left2, right2):
                if len(mask) >= mask_num:
                    break
                mask.add(i)
                spans[-1][-1] = i

    def ner_span_mask(self, 
                                input_text,
                                tokenizer, 
                                entity_spans=None,
                                return_only_spans=False,
                                ner_masking_prob=0.1,
                                mask_num=20,
                                **kargs):
        """mask tokens for masked language model training
        Args:
                sentence: 1d tensor, token list to be masked
                mask_ratio: ratio of tokens to be masked in the sentence
        Return:
                masked_sent: masked sentence
        """
        if not isinstance(input_text, list):
            sentence = tokenizer.encode(input_text, add_special_tokens=False)
            entity_spans = []
        else:
            sentence = input_text

        sent_length = len(sentence)

        mask = set()
        word_piece_map = self.paragraph_info.get_word_piece_map(sentence)
        spans = []
        
        while len(mask) < mask_num:
            if entity_spans:
                if np.random.random() <= ner_masking_prob:
                    self.mask_entity(sentence, mask_num, word_piece_map, spans, mask, entity_spans)
                else:
                    span_len = np.random.choice(self.lens, p=self.len_distrib)
                    anchor  = np.random.choice(sent_length)
                    if anchor in mask:
                        continue
                    self.mask_random_span(sentence, mask_num, word_piece_map, spans, mask, span_len, anchor)
            else:
                span_len = np.random.choice(self.lens, p=self.len_distrib)
                anchor  = np.random.choice(sent_length)
                if anchor in mask:
                    continue
                self.mask_random_span(sentence, mask_num, word_piece_map, spans, mask, span_len, anchor)
        sentence, target, pair_targets = mask_utils.span_masking(sentence, spans, self.tokens, self.pad, self.mask_id, self.max_pair_targets, mask, replacement=self.replacement_method, endpoints=self.endpoints)
        if return_only_spans:
            pair_targets = None
        return sentence, target, pair_targets
    
    def template_span_mask(self, 
                                input_text,
                                tokenizer, 
                                entity_spans=[],
                                return_only_spans=False,
                                ner_masking_prob=0.1,
                                mask_num=6,
                                ner_num=6,
                                **kargs):
        if not entity_spans:
            return self.ner_span_mask( 
                                input_text=input_text,
                                tokenizer=tokenizer, 
                                entity_spans=entity_spans,
                                return_only_spans=return_only_spans,
                                ner_masking_prob=ner_masking_prob,
                                mask_num=mask_num,
                                **kargs)
        
        if not isinstance(input_text, list):
            sentence = tokenizer.encode(input_text, add_special_tokens=False)
            entity_spans = []
        else:
            sentence = input_text

        sent_length = len(sentence)

        mask = set()
        word_piece_map = self.paragraph_info.get_word_piece_map(sentence)
        spans = []
        
        import random
        random.shuffle(entity_spans)
        for i in range(len(entity_spans)):
            self.mask_entity(sentence, mask_num, word_piece_map, spans, mask, entity_spans)
                    
        sentence, target, pair_targets = mask_utils.span_masking(sentence, spans, self.tokens, self.pad, self.mask_id, self.max_pair_targets, mask, replacement=self.replacement_method, endpoints=self.endpoints)
        if return_only_spans:
            pair_targets = None
        return sentence, target, pair_targets
    

