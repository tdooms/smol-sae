# import torch
# from torch import nn
# from einops import *
# from utils import ConstrainedAdam
# from base import BaseSAE, Loss

# class GatedSAE(BaseSAE):
#     """
#     A basic gated SAE implementation (no resampling).
#     https://arxiv.org/abs/2404.16014
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         device = config.device
        
#         # This is kinda scuffed but we need a slightly different encoder
#         del self.encoder

#         self.w_gate = nn.Parameter(self.decoder.weight.data.T.clone().to(device))
#         self.r_mag = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))

#         self.b_gate = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
#         self.b_mag = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        
#         self.optimizer = ConstrainedAdam(self.parameters(), [self.decoder.weight], lr=config.lr)

#     def encode(self, x):
#         preact = einsum(x - self.decoder.bias.data, self.w_gate, " ... inst d, inst h d -> ... inst h")
#         magnitude = preact * torch.exp(self.r_mag) + self.b_mag
        
#         hidden_act = torch.relu(magnitude) * (preact + self.b_gate > 0).float()
#         gated_act = torch.relu(preact + self.b_gate)

#         return hidden_act, gated_act

#     def decode(self, x):
#         return self.decoder(x)

#     def loss(self, x, _, x_hat, gated_act):
#         recons_losses = ((x - x_hat)**2).mean(dim=0).sum(dim=-1)

#         lambda_ = min(1, self.step / self.steps * 20)
#         sparsity_losses = gated_act.abs().mean(dim=0).sum(dim=-1)


#         gated_recons = self.decoder.detach()(gated_act)
#         aux_losses = (x - gated_recons).pow(2).mean(dim=0).sum(dim=-1)

#         return Loss(recons_losses, lambda_ * sparsity_losses, aux_losses)

#     def calculate_metrics(self, x_hid, losses, *args):
#         metrics = super().calculate_metrics(x_hid, losses, *args)
#         metrics |= {f"auxiliary_loss/{i}": l for i, l in enumerate(losses.auxiliary)}
#         return metrics
