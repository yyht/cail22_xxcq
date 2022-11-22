

from functools import partial
from operator import is_not

import torch
import random
import numpy as np

def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

def get_word2char_char2word(token_list, char_list):
    word_to_orig_index = []
    orig_to_word_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(token_list):
        orig_to_word_index.append(len(all_doc_tokens))
        for sub_token in list(token):
            word_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    word2char = []
    for index in range(len(orig_to_word_index)-1):
        word2char.append((orig_to_word_index[index], orig_to_word_index[index+1]))
    word2char.append((orig_to_word_index[-1], orig_to_word_index[-1]+1))

    char2word = [None] * len(char_list)
    for i, ((start, end)) in enumerate(word2char):
        char2word[start:end] = [i] * (end - start)
        
    return word2char, char2word

def char_span_to_word_span(char2word, char_span):
    token_indexes = char2word[char_span[0]:char_span[1]]
    token_indexes = list(filter(partial(is_not, None), token_indexes))
    if token_indexes:
        return token_indexes[0], token_indexes[-1] + 1  # [start, end)
    else:  # empty
        return 0, 0

def word_span_to_char_span(word2char, word_span):
    char_indexes = word2char[word_span[0]:word_span[1]]
    char_indexes = [span for span in char_indexes]  # 删除CLS/SEP对应的span
    start, end = char_indexes[0][0], char_indexes[-1][1]
    return start, end

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
    tokend = tokenizer(text, return_offsets_mapping=True, truncation=False, add_special_tokens=False)
    token2char = tokend.offset_mapping
    
    char2token = [None] * len(text)
    for i, ((start, end)) in enumerate(token2char):
        char2token[start:end] = [i] * (end - start)
    
    return token2char, char2token
