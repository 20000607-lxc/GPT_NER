# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# import copy_config as config
from .basic import BasicModule

class Attention(BasicModule):
    def __init__(self, emb_dim, hidden_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.dec_fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.up_hidden_dim = nn.Linear(hidden_dim//2, hidden_dim * 2)
        # if config.is_coverage:
        #     self.con_fc = nn.Linear(1, hidden_dim * 2, bias=False)
        self.init_params()

    def forward(self, s_t, enc_out, enc_padding_mask, coverage=None):
        """
        s_t : [batch_size, hidden_dim*2]
        enc_fea: [batch_size*sequence_length, hidden_dim*2]
        s_t and enc_fea to calculate the attention score

        enc_out: [batch_size, sequence_length, hidden_dim*2]
        """
        enc_out = self.up_hidden_dim(enc_out)# 扩大hidden dim: 从768到3072
        enc_fea = enc_out.view(-1, enc_out.shape[2])

        b, l, n = list(enc_out.size())

        dec_fea = self.dec_fc(s_t)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, l, n).contiguous()  # B x l x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)     # B*l x 2*hidden_dim

        att_features = enc_fea + dec_fea_expanded           # B*l x 2*hidden_dim
        # if config.is_coverage:
        #     coverage_inp = coverage.view(-1, 1)             # B*l x 1
        #     coverage_fea = self.con_fc(coverage_inp)        # B*l x 2*hidden_dim
        #     att_features = att_features + coverage_fea

        e = torch.tanh(att_features)                        # B*l x 2*hidden_dim
        scores = self.fc(e)                                 # B*l x 1
        scores = scores.view(-1, l)                         # B x l

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask  # B x l
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)                        # B x 1 x l   B x l x hidden dim
        c_t = torch.bmm(attn_dist, enc_out)                       # B x 1 x 2*hidden_dim
        c_t = c_t.view(-1, self.hidden_dim * 2)                   # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, l)                         # B x l

        # if config.is_coverage:
        #     coverage = coverage.view(-1, l)
        #     coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

