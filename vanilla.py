import torch
from torch import nn
from einops import *
from base import BaseSAE, Loss
from utils import ConstrainedAdam


class VanillaSAE(BaseSAE):
    """
    An ordinary SAE, no ghost grads or resampling.
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        device = config.device

        W_dec = torch.randn(self.n_instances, self.d_hidden, self.d_model, device=device)
        W_dec /= torch.norm(W_dec, dim=-1, keepdim=True) * 10
        self.W_dec = nn.Parameter(W_dec)

        W_enc = W_dec.mT.clone().to(device)
        self.W_enc = nn.Parameter(W_enc)

        self.b_enc = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(self.n_instances, self.d_model, device=device))

        self.relu = nn.ReLU()
        self.optimizer =  ConstrainedAdam(self.parameters(), [self.W_dec], lr=self.config.lr)

    def encode(self, x):
        return self.relu(einsum(x - self.b_dec, self.W_enc, "... inst d, inst d hidden -> ... inst hidden") + self.b_enc),

    def decode(self, h):
        return einsum(h, self.W_dec, "... inst hidden, inst hidden d -> ... inst d") + self.b_dec

    def loss(self, x, x_hid, x_hat, fraction):
        reconstruction = ((x_hat - x) ** 2).mean(0).sum(dim=-1)

        lambda_ = min(1, fraction * 20)
        sparsity = x_hid.abs().mean(dim=0).sum(-1)

        return Loss(reconstruction, lambda_ * sparsity, torch.zeros(self.n_instances, device=x.device))