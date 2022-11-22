
import torch
"""
https://github.com/redreamality/RERE-relation-extraction/blob/master/rere/extraction.py
"""

import numpy as np

def neg_sampling_unified(neg_mask, low=0.0, ratio=1.0):
    mask_shape = neg_mask.shape
    seed = torch.empty(mask_shape).uniform_(low, 1.0).to(neg_mask.device)  # generate a uniform random matrix with range [0, 1]
    seed *= (1-neg_mask)
    neg_mask_sample = torch.bernoulli(seed*ratio)
    return neg_mask_sample

def sample_labels(input_mask, input_labels, low=1.0, ratio=0.5, tril_mask=False):

    input_mask_tril = torch.unsqueeze(input_mask, dim=1).to(dtype=torch.float32) # [batch_size, 1, seq]
    input_mask_up = torch.unsqueeze(input_mask, dim=2).to(dtype=torch.float32) # [batch_size, 1, seq]
        
    target = torch.sum(input_labels, axis=1) # [batch_size, seq, seq]
    if tril_mask:
        valid_mask = torch.tril(torch.ones_like(target), diagonal=-1)
        valid_mask = 1.0 - (1.0 - valid_mask) * input_mask_tril
    else:
        valid_mask = input_mask_tril * input_mask_up
        valid_mask = 1.0 - valid_mask
    
    neg_mask = valid_mask.unsqueeze(dim=1) + input_labels # 1: for positive and 0-for negative logits
        
    # [batch_size, num_classes, seq, seq]
    none_neg_mask = neg_sampling_unified(neg_mask, low=low, ratio=ratio)    
    neg_mask_sample = none_neg_mask + input_labels
    
    return neg_mask_sample

def mid_loss(y_true, y_neg, y_pred, mid=0, pi=0.1, **kwargs):
    ## y_pred: sigmoid
    eps = 1e-6
    y_true = y_true.to(dtype=torch.float32) # [batch_size, 1, seq]
    pos = torch.sum(y_true * y_pred, dim=1) / (eps + torch.sum(y_true, dim=1))
    pos = - torch.log(pos + eps)
    neg = torch.sum(y_neg * y_pred, dim=1) / (eps + torch.sum(y_neg, dim=1))
    neg = torch.abs(neg-mid) 
    neg = - torch.log(1 - neg + eps)
    return torch.mean(pi*pos + neg)

def sampled_multilabel_categorical_crossentropy(y_true, y_pred, y_neg, threshold=0.0):
    y_pred = (1 - 2 * y_true) * y_pred
    
    y_pred_neg = y_pred - y_neg * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    threshold = torch.ones_like(y_pred[..., :1]) * threshold
    y_pred_neg = torch.cat([y_pred_neg, threshold], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, -threshold], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return torch.mean(neg_loss + pos_loss)