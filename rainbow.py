import torch
from torch import nn
from einops import *
from torch.optim import Adam
from base import BaseSAE, Loss

class RainbowSAE(BaseSAE):
    """
    This is a combination of gated SAE with the Anthropic loss.
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        device = config.device

        W_dec = torch.randn(self.n_instances, self.d_model, self.d_hidden, device=device)
        W_dec /= torch.norm(W_dec, dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)

        self.W_gate = nn.Parameter(W_dec.mT.clone())
        self.r_mag = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))

        self.b_gate = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_mag = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(self.n_instances, self.d_model, device=device))

        self.optimizer = Adam(self.parameters(), lr=config.lr, betas=(0.9, 0.999))

    def encode(self, x):
        preact = einsum(x, self.W_gate, "... inst d, inst h d -> ... inst h")
        magnitude = preact * torch.exp(self.r_mag) + self.b_mag
        
        hidden_act = torch.relu(magnitude) * (preact + self.b_gate > 0).float()
        gated_act = torch.relu(preact + self.b_gate)

        return hidden_act, gated_act

    def decode(self, x):
        return einsum(x, self.W_dec, "... inst h, inst d h -> ... inst d") + self.b_dec

    def loss(self, x, _, x_hat, gated_act):
        recons_losses = (x - x_hat).pow(2).mean(dim=0).sum(dim=-1)

        W_dec_clone = self.W_dec.detach()
        b_dec_clone = self.b_dec.detach()
        
        norm = W_dec_clone.norm(dim=-2)
        lambda_ = min(1, self.step/self.steps * 20)
        sparsity_losses = lambda_ * einsum(gated_act, norm, "batch inst h, inst h -> inst batch").mean(dim=-1)

        
        gated_recons = einsum(gated_act, W_dec_clone, "batch inst h, inst d h -> batch inst d") + b_dec_clone
        aux_losses = (x - gated_recons).pow(2).mean(dim=0).sum(dim=-1)

        return Loss(recons_losses, sparsity_losses, aux_losses)
    
    def calculate_metrics(self, x_hid, losses, *args):
        metrics = super().calculate_metrics(x_hid, losses, *args)
        metrics |= {f"auxiliary_loss/{i}": l for i, l in enumerate(losses.auxiliary)}
        return metrics