import torch
from torch import nn
from einops import *
from torch.optim import Adam
from base import BaseSAE, Loss


class AnthropicSAE(BaseSAE):
    """
    The main difference between this is the loss function.
    Specifically, it uses the activation * the output norm as the sparsity term.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        device = config.device

        W_dec = torch.randn(self.n_instances, self.d_model, self.d_hidden, device=device)
        W_dec /= torch.norm(W_dec, dim=-2, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.mT.clone())

        # Contrary to Anthropic, we actually still use a decoder norm because it seems more logical.
        self.b_enc = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(self.n_instances, self.d_model, device=device))

        self.optimizer = Adam(self.parameters(), lr=self.config.lr, betas=(0.9, 0.999))

    def encode(self, x):
        return torch.relu(einsum(x - self.b_dec, self.W_enc, "... inst d, inst h d -> ... inst h") + self.b_enc),

    def decode(self, x):
        return einsum(x, self.W_dec, "... inst h, inst d h -> ... inst d") + self.b_dec

    def loss(self, x, x_hid, x_hat):
        reconstruction = (x_hat - x).pow(2).mean(0).sum(dim=-1)

        norm = self.W_dec.norm(dim=-2)
        lambda_ = min(1, self.step/self.steps * 20)
        
        sparsity = einsum(x_hid, norm, "batch inst h, inst h -> batch inst").mean(dim=0)
        return Loss(reconstruction, lambda_ * sparsity, torch.zeros(self.n_instances, device=x.device))
