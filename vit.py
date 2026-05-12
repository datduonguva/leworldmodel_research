"""
Pytorch Implementation of ViT
"""

import torch
import torch.nn.functional as F
import numpy as np


PATCH_SIZE = 16
EMBEDDING_SIZE = 128


class TransformerLayer(torch.nn.Module):
    """ This implement a Multi-headed Self-Attention Layer"""
    def __init__(self, embedding_size: int, num_head: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.num_head = num_head
        assert (embedding_size % num_head) == 0
        self.head_size = int(embedding_size / num_head)
        self.layer_norm = torch.nn.LayerNorm(self.embedding_size)
        self.qkv_layer = torch.nn.Linear(
            in_features=self.embedding_size,
            out_features=self.embedding_size * 3
        )
        self.wo_projection = torch.nn.Linear(
            in_features=self.embedding_size,
            out_features=self.embedding_size,
        )

        self.mlp_0 = torch.nn.Linear(
            in_features=self.embedding_size,
            out_features=2 * self.embedding_size,
        )
        self.mlp_1 = torch.nn.Linear(
            in_features=2 *self.embedding_size,
            out_features=self. embedding_size,
        )
 

    def multiheaded_self_attention(self, x: torch.Tensor, mask=None): 
        B, L, D = x.shape
        # projection to (B, L, D)
        qkv = self.qkv_layer(x)  # (q, k, v)
        qkv = qkv.reshape((B, L, 3, self.num_head,  self.head_size))
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, N_h, L, D_h)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention

        # (B, N_h, L, L)
        attn_weights = torch.matmul(
            q, k.transpose(-1, -2)
        ) / self.head_size ** 0.5 
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # get the output -> B, N_h, L, D_h
        x = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, L, D)
        
        x = self.wo_projection(x)
        return x
    
    def forward(self, x: torch.Tensor, mask=None):
        """ Implementation of a MSA """
        
        x_normed = self.layer_norm(x)

        x = x + self.multiheaded_self_attention(x_normed, mask=mask)
        

        x = self.mlp_1(self.mlp_0(self.layer_norm(x))) + x
        return x
        
        

class ViT(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.unfold_layer = torch.nn.Unfold(
            kernel_size=PATCH_SIZE, stride=PATCH_SIZE
        )
        self.linear_project = torch.nn.Linear(
            in_features=3 * PATCH_SIZE * PATCH_SIZE,
            out_features=EMBEDDING_SIZE
        )

        self.positional_embeddings = torch.nn.Embedding(
            num_embeddings=int((224/ PATCH_SIZE)**2 + 1), # add 1 for CLS
            embedding_dim=EMBEDDING_SIZE
        )

        self.transformer_blocks = torch.nn.ModuleList([
            TransformerLayer(
                embedding_size=EMBEDDING_SIZE,
                num_head=4
            ) for l in range(4)
        ])
    

    def forward(self, x, training=False):
        
        # (B, C, H, W) -> (B , C*PATCH_SIZE *PATCH_SIZE, N_PATCHES)
        x = self.unfold_layer(x)
        x = torch.transpose(x, 1, 2)
        print("x shape after transpose: ", x.shape)
        B, N_PATCHES, D = x.shape

        # concat a cls token
        x = torch.cat((x, torch.zeros(B, 1, D)), dim=1)

        print("x shape after concat: ", x.shape)
        # linear projection
        x = self.linear_project(x)

        # add position encoding
        x = x + self.positional_embeddings(
            torch.arange(N_PATCHES + 1).unsqueeze(0)
        )
        
        print("shape after adding positional: ", x.shape) 

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        print("shape after transformer: ", x.shape) 

        return x

if __name__ == '__main__':
    x = torch.Tensor(np.random.rand(2, 3, 64, 64))
    print(x.shape)

    model = ViT()

    y = model(x)

    print(y.shape)
