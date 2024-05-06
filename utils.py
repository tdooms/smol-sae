from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from transformer_lens import utils

def get_splits(n_vals=32):
    training = load_dataset("NeelNanda/c4-tokenized-2b", split="train", streaming=True).with_format("torch")

    validation = list(training.take(n_vals))
    validation = torch.stack([row["tokens"] for row in validation])
    
    return training, validation


class ConstrainedAdam(torch.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    """
    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)
    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=-1, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=-1, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=-1, keepdim=True)
                
class Sampler:
    """
    A class for sampling activations from a model at a certain point in the network.
    It stores the activations in a large buffer and returns them in a single tensor.
    """
    def __init__(self, config, dataset, model):
        self.config = config
        self.model = model
        
        self.d_model = model.cfg.d_model
        self.n_ctx = model.cfg.n_ctx

        assert config.buffer_size % (config.in_batch * self.n_ctx) == 0, "samples must be a multiple of loader batch size"
        self.n_inputs = config.buffer_size // (config.in_batch * self.n_ctx)

        self.loader = DataLoader(dataset, batch_size=config.in_batch)
        self.batches = []

    def collect(self):
        result = rearrange(torch.cat(self.batches, dim=0), "... d_model -> (...) d_model")
        self.batches = []
        return result

    def extract(self, batch):
        hook_pt = utils.get_act_name(self.config.point, self.config.layer)
        _, cache = self.model.run_with_cache(batch, names_filter=[hook_pt], return_type="loss")
        return cache[hook_pt]

    @torch.inference_mode()
    def __iter__(self):
        self.batches = []

        for batch in self.loader:
            self.batches.append(self.extract(batch["tokens"]))
            
            if len(self.batches) == self.n_inputs:
                yield self.collect()