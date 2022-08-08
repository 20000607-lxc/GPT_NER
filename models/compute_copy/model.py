# -*- coding: utf-8 -*-
import torch
# import copy_config as config
from numpy import random
from .layers import Encoder
from .layers import Decoder
from .layers import ReduceState
#from transformer.model import TranEncoder

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class Model(object):
    def __init__(self, device, emb_dim, hidden_dim, model_path=None, is_eval=False, is_tran=False):
        self.device = device
        encoder = Encoder(emb_dim=emb_dim, hidden_dim=hidden_dim).to(self.device)
        decoder = Decoder(emb_dim=emb_dim, hidden_dim=hidden_dim, device=device).to(self.device)
        reduce_state = ReduceState(hidden_dim=hidden_dim).to(self.device)
        # if is_tran:
        #     encoder = TranEncoder(config.vocab_size, config.max_enc_steps, config.emb_dim,
        #          config.n_layers, config.n_head, config.d_k, config.d_v, config.d_model, config.d_inner)
        # shared the embedding between encoder and decoder
        # decoder.tgt_word_emb.weight = encoder.src_word_emb.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()
        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state
        # if model_path is not None:
        #     state = torch.load(model_path, map_location=lambda storage, location: storage)
        #     self.encoder.load_state_dict(state['encoder_state_dict'])
        #     self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
        #     self.reduce_state.load_state_dict(state['reduce_state_dict'])
