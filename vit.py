"""
Pytorch Implementation of ViT
"""

import torch
import numpy as np


PATCH_SIZE = 16
EMBEDDING_SIZE = 128

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

        self.positional_embeddings = torch.nn.Linear
            in_features=(224/ PATCH_SIZE)**2 + 1, # add 1 for CLS
            out_features=EMBEDDING_SIZE
        )

    def forward(self, x, training=False):
        
        # (B, C, H, W) -> (B , C*PATCH_SIZE *PATCH_SIZE, N_PATCHES)
        x = self.unfold_layer(x)
        x = torch.transpose(x, 1, 2)

        # (B, n_patches, large_dim) -> (B, n_patches, EMBEDDING_SIZE)
        x = self.linear_project(x)

        # add position encoding
        # for image -> position is (0, 1, 2, 3, 4)
        # positional emebddings is (L, D) (B, L)
        
        return x

if __name__ == '__main__':
    x = torch.Tensor(np.random.rand(2, 3, 64, 64))
    print(x.shape)

    model = ViT()

    y = model(x)

    print(y.shape)
