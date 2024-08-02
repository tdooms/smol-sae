import torch
from torch import nn
from einops import *
from torch.optim import Adam
from base import BaseSAE


class NormedSAE(BaseSAE):
    """
    The main difference between this is the loss function.
    Specifically, it uses the activation * the output norm as the sparsity term.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """
    def __init__(self, config):
        super().__init__(config)
        self.optimizer = Adam(self.parameters(), lr=self.config.lr, betas=(0.9, 0.999))

    def encode(self, x):
        return torch.relu(self.encoder(x - self.decoder.bias.data))

    def loss(self, x, x_hid, x_hat):
        reconstruction = (x_hat - x).pow(2).mean(0).sum(dim=-1)

        norm = self.decoder.weight.data.norm(dim=-2)
        lambda_ = min(1, self.step / self.steps * 20)
        
        sparsity = einsum(x_hid, norm, "batch inst h, inst h -> batch inst").mean(dim=0)
        return reconstruction + lambda_ * sparsity
