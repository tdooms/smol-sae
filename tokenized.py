import torch
from topk import TopkSAE
from torch.optim import Adam
from torch import nn
from utils import create_unigram_lookup_table

class TokenizedSAE(TopkSAE):
    """
    The top-k SAE, augmented with a per-token bias.
    """
    def __init__(self, config, model):
        super().__init__(config)
        
        # Halve the scale of both the lookup table and the encoder to balance their effect
        original = 0.5 * create_unigram_lookup_table(model, config).detach()
        self.lookup = nn.Embedding.from_pretrained(original).to(config.device)
        
        self.encoder.weight.data *= 0.5
        
        # Overwrite the optimizer to include the lookup table with a slightly higher learning rate.
        parameters = [
            dict(params=list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.config.lr),
            dict(params=self.lookup.parameters(), lr=0.01)
        ]
        
        self.optimizer = Adam(parameters, betas=(0.9, 0.999))
    
    def decode(self, x, y):
        # Add the lookup table to the decoder output
        return self.decoder(x) + self.lookup(y)

