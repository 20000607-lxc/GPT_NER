# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from .basic import BasicModule
from .attention import Attention


class Decoder(BasicModule):
    def __init__(self, emb_dim, hidden_dim, device):
        super(Decoder, self).__init__()
        self.attention_network = Attention(emb_dim, hidden_dim)
        self.con_fc = nn.Linear(hidden_dim * 2 + emb_dim, emb_dim)
        self.con_fc2 = nn.Linear(hidden_dim * 2, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.device = device

        if True: #config.pointer_gen:
            self.p_gen_fc = nn.Linear(hidden_dim * 4 + emb_dim, 1)
        # p_vocab
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.init_params()

    def forward(self, y_t_embd, no_teacher_force, s_t, enc_out,  enc_padding_mask,
                c_t, step, vocab_dist, enc_batch_extend_vocab=None, coverage=None):
        if not self.training and step == 0:
            dec_h, dec_c = s_t
            s_t_hat = torch.cat((dec_h.view(-1, self.hidden_dim),
                                 dec_c.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, enc_out,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        if step == 0:
            # the first step, y_t_embd is the begin token id embedding ( currently choose '  ' as the begin token)
            #b = torch.cat((c_t, y_t_embd), 1)# 2*hidden + emb
            x = self.con_fc2(c_t)

            # begin_id_embeddings = torch.empty(len(c_t), self.emb_dim).to(self.device)
            # for k in range(len(c_t)):
            #     begin_id_embeddings[k] = begin_id_embedding
            # do not use the begin_id_embeddings here! for it causes loss_backwards problem which is not fixed yet!

        else:
            x = torch.empty(len(c_t), self.emb_dim).to(self.device)
            # if the label id (entity id) is -100
            # (-100 means the entity is already ended) then do not add id.
            for i in range(len(c_t)):
                if no_teacher_force[i]:
                    x[i] = self.con_fc2(c_t[i])
                else:
                    b = torch.cat((c_t[i], y_t_embd[i]), 0)# 2*hidden + emb
                    x[i] = self.con_fc(b)

        _, s_t = self.lstm(x.unsqueeze(1), s_t)# add (teacher forcing: y_t_emb) and (former step context c_t) into s_t
        dec_h, dec_c = s_t
        s_t_hat = torch.cat((dec_h.view(-1, self.hidden_dim),
                             dec_c.view(-1, self.hidden_dim)), 1)     # B x 2*hidden_dim = 4*768
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, enc_out,
                                                               enc_padding_mask, coverage)
        if self.training or step > 0:
            coverage = coverage_next
        p_gen_inp = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
        p_gen = self.p_gen_fc(p_gen_inp)
        p_gen = torch.sigmoid(p_gen)
        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist# [bz, sequence_length]
        final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)#[bz, extend_vocab_size]
        return final_dist, s_t, c_t, attn_dist, p_gen, coverage
