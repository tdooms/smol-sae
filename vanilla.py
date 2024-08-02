import torch
from base import BaseSAE
from utils import ConstrainedAdam

class VanillaSAE(BaseSAE):
    """
    An ordinary SAE, no ghost grads or resampling.
    """
    def __init__(self, config):
        super().__init__(config)
        self.optimizer =  ConstrainedAdam(self.parameters(), [self.decoder.weight], lr=self.config.lr)

    def encode(self, x):
        return torch.relu(self.encoder(x - self.decoder.bias.data))

    def loss(self, x, x_hid, x_hat):
        reconstruction = (x_hat - x).pow(2).mean(0).sum(dim=-1)

        # Calculate the sparsity loss and its warmup factor
        lambda_ = min(1, self.step/self.steps * 20)
        sparsity = x_hid.abs().mean(dim=0).sum(-1)

        return reconstruction + lambda_ * sparsity