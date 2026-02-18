# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from sam3.backbones.mobile_clip import MobileCLIPTextTransformer
from sam3.backbones.cnn_text import CNNTextBackbone
from sam3.model.tokenizer_ve import SimpleTokenizer

class StudentTensorRunner(nn.Module):
    """Helper module to group tensor operations for compilation."""
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, tokenized):
        # 1. Embed
        input_embeds = self.encoder.forward_embedding(tokenized)
        
        # 2. Transform
        # Use embeddings as input to skip redundant embedding lookup
        text_memory = self.encoder(
            input_embeds, 
            return_all_tokens=True, 
            input_is_embeddings=True
        )
        
        # 3. Project
        text_memory = self.projector(text_memory)
        
        return text_memory, input_embeds

class TextStudentEncoder(nn.Module):
    def __init__(self, cfg, context_length, output_dim, bpe_path=None):
        super().__init__()
        self.context_length = context_length
        
        if bpe_path is None:
            bpe_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
            )
        
        self.tokenizer = SimpleTokenizer(bpe_path=bpe_path)
        
        # MobileCLIP Transformer (Student)
        encoder = MobileCLIPTextTransformer(
            cfg=cfg,
            projection_dim=cfg["dim"]
        )
        
        # Post-Projection (Student Dim -> Output Dim)
        projector = nn.Linear(cfg["dim"], output_dim)
        
        self.tensor_runner = StudentTensorRunner(encoder, projector)

        # Expose encoder for checkpoint loading compatibility
        # Checkpoints expect 'encoder' and 'projector' attributes
        self.encoder = self.tensor_runner.encoder
        self.projector = self.tensor_runner.projector

    def set_context_length(self, context_length: int):
        """Set the text encoder context length and resize positional embeddings if needed.
        
        Args:
            context_length (int): New context length
        """
        self.context_length = context_length
        if hasattr(self.encoder, "resize_pos_embed"):
            self.encoder.resize_pos_embed(context_length)

    def forward(self, text, input_boxes=None, device=None):
        # 1. Tokenize (CPU/Eager)
        tokenized = self.tokenizer(text, context_length=self.context_length).to(device)
        
        # 2. Run Tensor Operations (Compilable)
        text_memory, input_embeds = self.tensor_runner(tokenized)
        
        # 3. Prepare output
        text_attention_mask = (tokenized != 0).bool().ne(1)
        
        return text_attention_mask, text_memory.transpose(0, 1), input_embeds.transpose(0, 1)

class CNNTextStudentEncoder(nn.Module):
    def __init__(self, cfg, context_length, output_dim, bpe_path=None):
        super().__init__()
        self.context_length = context_length
        
        if bpe_path is None:
            # Try to locate BPE file relative to this file
            bpe_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
            )
        
        self.tokenizer = SimpleTokenizer(bpe_path=bpe_path)
        
        # Ensure cfg has context_length
        cfg["context_length"] = context_length
        
        encoder = CNNTextBackbone(cfg)
        
        projector = nn.Linear(cfg["dim"], output_dim)
        
        self.tensor_runner = StudentTensorRunner(encoder, projector)

        self.encoder = self.tensor_runner.encoder
        self.projector = self.tensor_runner.projector

    def forward(self, text, input_boxes=None, device=None):
        tokenized = self.tokenizer(text, context_length=self.context_length).to(device)
        text_memory, input_embeds = self.tensor_runner(tokenized)
        text_attention_mask = (tokenized != 0).bool().ne(1)
        return text_attention_mask, text_memory.transpose(0, 1), input_embeds.transpose(0, 1)
