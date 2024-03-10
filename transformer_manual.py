""" Based on https://github.com/TranQuocTrinh/transformer """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import math
import numpy as np
import random

# embedding_dim == n_prunable_layers (106)
# dim_model == n_features (2)


# The positional encoding vector, embedding_dim is d_model
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_length, embedding_dim)
        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / embedding_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i + 1) / embedding_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_dim)
        seq_length = x.size(1)
        pe = Variable(self.pe[:, :seq_length], requires_grad=False).to(x.device)
        # Add the positional encoding vector to the embedding vector
        x = x + pe
        x = self.dropout(x)
        return x.permute(1, 0, 2)

"""
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        print(f"PositionalEncoder forward: {token_embedding.shape}, {self.pos_encoding[:token_embedding.size(0), :].shape} ")
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :]).permute(1, 0, 2)
"""

class Transformer(nn.Module):
    def __init__(self, nhead, nlayers, dim_model, dim_ff,  out_size, dropout, norm=True):
        super(Transformer, self).__init__()
        self.initial_linear = nn.Linear(6, dim_model)
        self.positional_encoder = PositionalEncoder(embedding_dim=dim_model)
        self.__encoder_layer = nn.TransformerEncoderLayer(dim_model, nhead, dim_feedforward=dim_ff, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.__encoder_layer, num_layers=nlayers, norm=nn.LayerNorm(dim_model))
        self.final_linear = nn.Linear(dim_model, out_size)

    def forward(self, src, src_mask=None):
        src = self.initial_linear(src)
        # print(f"src in transformer forward before PosEnd: {src.shape}")
        x = self.positional_encoder(src)
        # print(f"x in transformer forward after PosEnd: {x.shape}")
        x = self.encoder(x)
        #x = self.encoder(x)
        # print(f"x in transformer forward after Encoder: {x.shape}")
        out = self.final_linear(x[0, :, :])
        # out = self.final_linear(x)

        return out

def get_tgt_mask(size) -> torch.tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]

    return mask