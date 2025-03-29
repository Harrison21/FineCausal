from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class CausalAttention(nn.Module):
    def __init__(self, embed_size=1024, n_heads=4, num_layers=2):
        super(CausalAttention, self).__init__()

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=True,  # Enable causal masking
                            factor=1,
                            attention_dropout=0.1,
                            output_attention=True,
                        ),
                        embed_size,
                        n_heads,
                    ),
                    embed_size,
                    256,
                    dropout=0.01,
                    activation="gelu",
                )
                for l in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(embed_size),
        )

    def forward(self, combined):
        # combined: video_1_fea_re from helper.py [2, 12, 1024]

        # Apply causal attention
        output, attention = self.encoder(combined)

        return output, attention
