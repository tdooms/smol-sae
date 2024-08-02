import torch
from base import BaseSAE
from torch.optim import Adam

class TopkSAE(BaseSAE):
    """
    The top-k SAE. 
    """
    def __init__(self, config):
        super().__init__(config)
        self.optimizer = Adam(self.parameters(), lr=self.config.lr, betas=(0.9, 0.999))
        assert isinstance(self.config.sparsity, int), "Sparsity must be an integer (which represents k)"

    def encode(self, x):
        x_hid = self.encoder(x - self.decoder.bias)
        indices = x_hid.topk(k=self.config.sparsity, dim=-1).indices

        mask = torch.zeros_like(x_hid)
        mask.scatter_(-1, indices, 1)
        
        return x_hid * mask

    def loss(self, x, x_hid, x_hat):
        return (x_hat - x).pow(2).mean(0).sum(dim=-1)