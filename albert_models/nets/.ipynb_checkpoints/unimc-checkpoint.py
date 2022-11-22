
import torch
from torch import nn
import json
from tqdm import tqdm
import os
import numpy as np

from transformers import BertForMaskedLM

class UniMCModel(nn.Module):
    def __init__(self, encoder, yes_token):
        super().__init__()
        self.encoder = encoder
        # self.config = AutoConfig.from_pretrained(pre_train_dir)
        # if self.config.model_type == 'megatron-bert':
        #     self.bert = MegatronBertForMaskedLM.from_pretrained(pre_train_dir)
        # elif self.config.model_type == 'deberta-v2':
        #     self.bert = DebertaV2ForMaskedLM.from_pretrained(pre_train_dir)
        # elif self.config.model_type == 'albert':
        #     self.bert = AlbertForMaskedLM.from_pretrained(pre_train_dir)
        # else:
        #     self.bert = BertForMaskedLM.from_pretrained(pre_train_dir)

        self.loss_func = torch.nn.CrossEntropyLoss()
        self.yes_token = yes_token

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids=None, mlmlabels=None, clslabels=None, clslabels_mask=None, mlmlabels_mask=None):

        batch_size, seq_len = input_ids.shape
        outputs = self.encoder(input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            labels=mlmlabels)  # (bsz, seq, dim)
        mask_loss = outputs.loss
        mlm_logits = outputs.logits
        cls_logits = mlm_logits[:, :,
                                self.yes_token].view(-1, seq_len)+clslabels_mask

        if mlmlabels == None:
            return 0, mlm_logits, cls_logits
        else:
            cls_loss = self.loss_func(cls_logits, clslabels)
            all_loss = mask_loss+cls_loss
            return all_loss, mlm_logits, cls_logits