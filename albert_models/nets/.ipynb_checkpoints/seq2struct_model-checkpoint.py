
import torch
import io
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import datetime
import torch.nn as nn
import logging
import warnings

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training
    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module
        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob
        
class MyUniLM(nn.Module):
    def __init__(self, config_path, model_path, eos_token_id, **kargs):
        super().__init__()
        
        from nets.unilm_bert import BertForCausalLM
        from transformers import BertConfig

        self.model_path = model_path
        self.config_path = config_path
        self.eos_token_id = eos_token_id
        
        self.config = BertConfig.from_pretrained(self.config_path)
        self.config.is_decoder = True
        self.config.eos_token_id = self.eos_token_id
        
        self.transformer = BertForCausalLM.from_pretrained(
                                pretrained_model_name_or_path=self.model_path,
                                config=self.config)
        
    def forward(self, input_ids, input_mask, segment_ids=None, mode='train', **kargs):
        if mode == "train":
            idxs = torch.cumsum(segment_ids, dim=1)
            attention_mask_3d = (idxs[:, None, :] <= idxs[:, :, None]).to(dtype=torch.float32)
            model_outputs = self.transformer(input_ids, 
                                             attention_mask=attention_mask_3d, 
                                             token_type_ids=segment_ids)
            return model_outputs # return prediction-scores
        elif mode == "generation":
            model_outputs = self.transformer.generate(
                                            input_ids=input_ids, 
                                            attention_mask=input_mask, 
                                            token_type_ids=segment_ids, 
                                            **kargs) # we need to generate output-scors
            
class MyMengziT5(nn.Module):
    def __init__(self, config_path, model_path, **kargs):
        super().__init__()
        from transformers import T5ForConditionalGeneration, T5Config
        self.config = T5Config.from_pretrained(config_path)
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path,
                                                                config=self.config)
        
    def forward(self, input_ids, input_mask, decoder_input_ids=None, mode='train', **kargs):
        
        if mode == "train":
            model_outputs = self.model(input_ids=input_ids, attention_mask=input_mask,
                      decoder_input_ids=decoder_input_ids)
        elif mode == "generation":
             model_outputs = self.transformer.generate(
                                            input_ids=input_ids, 
                                            attention_mask=input_mask, 
                                            token_type_ids=segment_ids, 
                                            **kargs) # we need to generate output-scors
        return model_outputs