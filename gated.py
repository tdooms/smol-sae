import torch
from torch import nn
from einops import *
from utils import ConstrainedAdam
from base import BaseSAE, Loss

class GatedSAE(BaseSAE):
    """
    A basic gated SAE implementation (no resampling).
    https://arxiv.org/abs/2404.16014
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        device = config.device

        W_dec = torch.randn(self.n_instances, self.d_hidden, self.d_model, device=device)
        W_dec /= torch.norm(W_dec, dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)

        self.W_gate = nn.Parameter(W_dec.mT.clone().to(device))
        self.r_mag = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))

        self.b_gate = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_mag = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(self.n_instances, self.d_model, device=device))

        self.relu = nn.ReLU()
        self.optimizer = ConstrainedAdam(self.parameters(), [self.W_dec], lr=config.lr)

    def encode(self, x):
        gated_preact = einsum(self.W_gate, x - self.b_dec, "inst d h, ... inst d -> ... inst h") + self.b_gate
        gate = (gated_preact > 0).float()

        W_mag = einsum(torch.exp(self.r_mag), self.W_gate, "inst h, inst d h -> inst d h")
        magnitude = einsum(W_mag, x - self.b_dec, "inst d h, ... inst d -> ... inst h") + self.b_mag
        mag_act = torch.relu(magnitude)
        hidden_act = mag_act * gate

        gated_act = torch.relu(gated_preact)

        return hidden_act, gated_act

    def decode(self, h):
        return einsum(h, self.W_dec, "... inst h, inst h d -> ... inst d") + self.b_dec

    def loss(self, x, _, x_hat, fraction, gated_act):
        recons_losses = ((x - x_hat)**2).mean(dim=0).sum(dim=-1)

        lambda_ = min(1, fraction * 20)
        sparsity_losses = gated_act.abs().mean(dim=0).sum(dim=-1)

        W_dec_clone = self.W_dec.clone().detach()
        b_dec_clone = self.b_dec.clone().detach()

        gated_recons = einsum(gated_act, W_dec_clone, "batch inst hidden, inst hidden d -> batch inst d") + b_dec_clone
        aux_losses = ((x - gated_recons)**2).mean(dim=0).sum(dim=-1)

        return Loss(recons_losses, lambda_ * sparsity_losses, aux_losses)
    
    def calculate_metrics(self, x_hid, losses, *args):
        metrics = super().calculate_metrics(x_hid, losses, *args)
        metrics |= {f"auxiliary_loss/{i}": l for i, l in enumerate(losses.auxiliary)}
        return metrics