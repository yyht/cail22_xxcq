
import numpy as np
import unicodedata, re
from copy import deepcopy

def convert(set):
    return sorted(set)

def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0, 1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def read_conll_data(data_path, mode="train"):
    datalist = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines.append('\n')

        text = []
        labels = []
        label_set = set()

        for line in lines: 
            if line == '\n':                
                text = ''.join(text)
                entity_labels = []
                for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                    entity_labels.append({
                        'start_idx': _start_idx,
                        'end_idx': _end_idx,
                        'type': _type,
                        'entity': text[_start_idx: _end_idx+1]
                    })

                if text == '':
                    continue

                datalist.append({
                    'text': text,
                    'label': entity_labels
                })

                text = []
                labels = []

            elif line == '  O\n':
                text.append(' ')
                labels.append('O')
            else:
                line = line.strip('\n').split()
                if len(line) == 1:
                    term = ' '
                    label = line[0]
                else:
                    term, label = line
                text.append(term)
                label_set.add(label.split('-')[-1])
                labels.append(label)
    return datalist
    
def _is_control(ch):
    """控制类字符判断
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')

def _is_special(ch):
    """判断是不是有特殊含义的符号
    """
    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

def recover_bert_token(token):
    """获取token的“词干”（如果是##开头，则自动去掉##）
    """
    if token[:2] == '##':
        return token[2:]
    else:
        return token

def get_token_mapping(text, tokens, additional_special_tokens=set(), is_mapping_index=True):
    """给出原始的text和tokenize后的tokens的映射关系"""
    raw_text = deepcopy(text)
    text = text.lower()

    normalized_text, char_mapping = '', []
    for i, ch in enumerate(text):
        ch = unicodedata.normalize('NFD', ch)
        ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
        ch = ''.join([
            c for c in ch
            if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))
        ])
        normalized_text += ch
        char_mapping.extend([i] * len(ch))

    text, token_mapping, offset = normalized_text, [], 0
    for token in tokens:
        token = token.lower()
        if token == '[unk]' or token in additional_special_tokens:
            if is_mapping_index:
                token_mapping.append(char_mapping[offset:offset+1])
            else:
                token_mapping.append(raw_text[offset:offset+1])
            offset = offset + 1
        elif _is_special(token):
            # 如果是[CLS]或者是[SEP]之类的词，则没有对应的映射
            token_mapping.append([])
        else:
            token = recover_bert_token(token)
            start = text[offset:].index(token) + offset
            end = start + len(token)
            if is_mapping_index:
                token_mapping.append(char_mapping[start:end])
            else:
                token_mapping.append(raw_text[start:end])
            offset = end

    return token_mapping

def from_token_mapping2label(num_classes, max_len, label_list, token_mapping):
    labels = np.zeros((num_classes, max_len, max_len))
    
    start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j}
    end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j}

    for item in label_list:
        start = item['start_idx']
        end = item['end_idx']
        label = item['type']
        if start in start_mapping and end in end_mapping:
            start = start_mapping[start]
            end = end_mapping[end]
            label = int(label)
            labels[label, start, end] = 1
            
    return labels

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

import sys
import re
import io

LHan = [
  [0x2E80, 0x2E99],  # Han # So  [26] CJK RADICAL REPEAT, CJK RADICAL RAP
  [0x2E9B, 0x2EF3
   ],  # Han # So  [89] CJK RADICAL CHOKE, CJK RADICAL C-SIMPLIFIED TURTLE
  [0x2F00, 0x2FD5],  # Han # So [214] KANGXI RADICAL ONE, KANGXI RADICAL FLUTE
  0x3005,  # Han # Lm       IDEOGRAPHIC ITERATION MARK
  0x3007,  # Han # Nl       IDEOGRAPHIC NUMBER ZERO
  [0x3021,
   0x3029],  # Han # Nl   [9] HANGZHOU NUMERAL ONE, HANGZHOU NUMERAL NINE
  [0x3038,
   0x303A],  # Han # Nl   [3] HANGZHOU NUMERAL TEN, HANGZHOU NUMERAL THIRTY
  0x303B,  # Han # Lm       VERTICAL IDEOGRAPHIC ITERATION MARK
  [
    0x3400, 0x4DB5
  ],  # Han # Lo [6582] CJK UNIFIED IDEOGRAPH-3400, CJK UNIFIED IDEOGRAPH-4DB5
  [
    0x4E00, 0x9FC3
  ],  # Han # Lo [20932] CJK UNIFIED IDEOGRAPH-4E00, CJK UNIFIED IDEOGRAPH-9FC3
  [
    0xF900, 0xFA2D
  ],  # Han # Lo [302] CJK COMPATIBILITY IDEOGRAPH-F900, CJK COMPATIBILITY IDEOGRAPH-FA2D
  [
    0xFA30, 0xFA6A
  ],  # Han # Lo  [59] CJK COMPATIBILITY IDEOGRAPH-FA30, CJK COMPATIBILITY IDEOGRAPH-FA6A
  [
    0xFA70, 0xFAD9
  ],  # Han # Lo [106] CJK COMPATIBILITY IDEOGRAPH-FA70, CJK COMPATIBILITY IDEOGRAPH-FAD9
  [
    0x20000, 0x2A6D6
  ],  # Han # Lo [42711] CJK UNIFIED IDEOGRAPH-20000, CJK UNIFIED IDEOGRAPH-2A6D6
  [0x2F800, 0x2FA1D]
]  # Han # Lo [542] CJK COMPATIBILITY IDEOGRAPH-2F800, CJK COMPATIBILITY IDEOGRAPH-2FA1D

CN_PUNCTS = [(0x3002, "。"), (0xFF1F, "？"), (0xFF01, "！"), (0xFF0C, "，"),
       (0x3001, "、"), (0xFF1B, "；"), (0xFF1A, "："), (0x300C, "「"),
       (0x300D, "」"), (0x300E, "『"), (0x300F, "』"), (0x2018, "‘"),
       (0x2019, "’"), (0x201C, "“"), (0x201D, "”"), (0xFF08, "（"),
       (0xFF09, "）"), (0x3014, "〔"), (0x3015, "〕"), (0x3010, "【"),
       (0x3011, "】"), (0x2014, "—"), (0x2026, "…"), (0x2013, "–"),
       (0xFF0E, "．"), (0x300A, "《"), (0x300B, "》"), (0x3008, "〈"),
       (0x3009, "〉"), (0x2015, "―"), (0xff0d, "－"), (0x0020, " ")]
#(0xFF5E, "～"),

EN_PUNCTS = [[0x0021, 0x002F], [0x003A, 0x0040], [0x005B, 0x0060],
       [0x007B, 0x007E]]


class ChineseAndPunctuationExtractor(object):
  def __init__(self):
    self.chinese_re = self.build_re()

  def is_chinese_or_punct(self, c):
    if self.chinese_re.match(c):
      return True
    else:
      return False

  def build_re(self):
    L = []
    for i in LHan:
      if isinstance(i, list):
        f, t = i
        try:
          f = chr(f)
          t = chr(t)
          L.append('%s-%s' % (f, t))
        except:
          pass  # A narrow python build, so can't use chars > 65535 without surrogate pairs!

      else:
        try:
          L.append(chr(i))
        except:
          pass
    for j, _ in CN_PUNCTS:
      try:
        L.append(chr(j))
      except:
        pass

    for k in EN_PUNCTS:
      f, t = k
      try:
        f = chr(f)
        t = chr(t)
        L.append('%s-%s' % (f, t))
      except:
        raise ValueError()
        pass  # A narrow python build, so can't use chars > 65535 without surrogate pairs!

    RE = '[%s]' % ''.join(L)
    # print('RE:', RE.encode('utf-8'))
    return re.compile(RE, re.UNICODE)

import re
import unicodedata
extractor = ChineseAndPunctuationExtractor()

def is_whitespace(c):
  if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    return True
  return False

def flat_list(h_list):
  e_list = []

  for item in h_list:
    if isinstance(item, list):
      e_list.extend(flat_list(item))
    else:
      e_list.append(item)
  return e_list

def text_tokenize(tokenizer, text_raw):
  sub_text = []
  buff = ""
  flag_en = False
  flag_digit = False
  for char in text_raw:
    if extractor.is_chinese_or_punct(char):
      if buff != "":
        sub_text.append(buff)
        buff = ""
      sub_text.append(char)
      flag_en = False
      flag_digit = False
    else:
      if re.compile('\d').match(char):
        if buff != "" and flag_en:
          sub_text.append(buff)
          buff = ""
          flag_en = False
        flag_digit = True
        buff += char
      else:
        if buff != "" and flag_digit:
          sub_text.append(buff)
          buff = ""
          flag_digit = False
        flag_en = True
        buff += char

  if buff != "":
    sub_text.append(buff)
  all_doc_tokens = []
  for (i, token) in enumerate(sub_text):
    sub_tokens = tokenizer.tokenize(token)
    for sub_token in sub_tokens:
      all_doc_tokens.append(sub_token)
      
  return all_doc_tokens