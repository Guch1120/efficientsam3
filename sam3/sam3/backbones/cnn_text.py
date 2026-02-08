# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class CNNTextBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg.get("vocab_size", 49408)
        self.dim = cfg.get("dim", 384)
        self.context_length = cfg.get("context_length", 16)
        self.num_layers = cfg.get("n_transformer_layers", 4) # re-using config name for consistency
        self.kernel_size = cfg.get("kernel_size", 3)
        self.dropout = cfg.get("embed_dropout", 0.0)
        
        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        torch.nn.init.normal_(self.embedding.weight, std=0.02)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.context_length, self.dim))
        trunc_normal_(self.pos_embed, std=0.02)
        
        self.layers = nn.ModuleList()
        # Using a simple ResNet-1D style block
        for _ in range(self.num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(self.dim, self.dim, self.kernel_size, padding=self.kernel_size//2),
                    nn.BatchNorm1d(self.dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Conv1d(self.dim, self.dim, self.kernel_size, padding=self.kernel_size//2),
                    nn.BatchNorm1d(self.dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout)
                )
            )

    def forward_embedding(self, tokenized):
        x = self.embedding(tokenized) # B, L, D
        seq_len = x.shape[1]
        # Handle Positional Embedding
        # If seq_len > context_length, it naturally fails or we should truncate/interpolate?
        # Tokenizer ensures length <= context_length usually.
        # But if we padded, seq_len is context_length.
        if seq_len <= self.context_length:
             x = x + self.pos_embed[:, :seq_len, :]
        return x

    def forward(self, x, return_all_tokens=True, input_is_embeddings=False):
        # x is (B, L, D) embeddings
        x = x.transpose(1, 2) # (B, D, L)
        
        for layer in self.layers:
            residual = x
            out = layer(x)
            x = out + residual
            
        x = x.transpose(1, 2) # (B, L, D)
        return x
