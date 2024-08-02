from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from transformers import DataCollatorForLanguageModeling
import torch
import gc


def get_dataset_splits(tokenizer, n_validation=32):
    tokenize = lambda batch: tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
    
    training = load_dataset("c4", 'en', split="train", streaming=True).with_format("torch")
    training = training.map(tokenize, batched=True, remove_columns=["text", "url", "timestamp"])

    validation = list(training.take(n_validation))
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    validation = collator(validation)
    
    return training, validation


@torch.no_grad()
def create_unigram_lookup_table(model, config):
    hook_pt = config.hook_pt
    vocab_size = len(model.tokenizer.vocab) # obviously this isn't he same as tokenizer.vocab_size, duh
    batch_size = 1024
    
    tokens = torch.arange(vocab_size, device=config.device, dtype=torch.long)
    bosses = torch.ones_like(tokens) * model.tokenizer.bos_token_id
    
    full = torch.stack([bosses, tokens], dim=-1)
    result = torch.empty(vocab_size, model.cfg.d_model, device=config.device)
    
    for batch in full.split(batch_size, dim=0):
        _, cache = model.run_with_cache(batch, names_filter=[hook_pt], stop_at_layer=config.layer + 1)
        result[batch[:, 1]] = cache[hook_pt][:, -1]
    
    return result.detach()


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
    

class TokenDataset(Dataset):
    """This class is a dynamic dataset that samples activations from a model on the fly."""
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        
        self.d_model = model.cfg.d_model
        self.n_ctx = model.cfg.n_ctx
        self.n_tokens = config.n_tokens

        assert config.buffer_size % (config.in_batch * self.n_ctx) == 0, "samples must be a multiple of loader batch size"
        self.n_inputs = config.buffer_size // (config.in_batch * self.n_ctx)

        # This is somewhat of a hack but using the iter object retains the state throughout for loops.
        # If we were to use the dataloader immediately, it would sample the same data over and over.
        self.loader = DataLoader(dataset, batch_size=config.in_batch)
        self.iter = iter(self.loader)
        
        self.start, self.end = 0, 0
    
    @torch.no_grad()
    def collect(self):
        activations, tokens = [], []
        
        for _, batch in zip(range(self.n_inputs), self.iter):
            # Make sure we only sample the first n tokens
            inputs = batch["input_ids"][..., :self.n_ctx]
            activations.append(self.extract(inputs))
            tokens.append(inputs.to(self.config.device))
        
        activations = rearrange(torch.cat(activations, dim=0), "... d_model -> (...) d_model")
        tokens = torch.cat(tokens, dim=0).flatten()
        
        # Shuffle the activations and tokens
        # perm = torch.randperm(len(tokens))
        # self.activations, self.tokens = activations[perm], tokens[perm]
        self.activations, self.tokens = activations, tokens

        # This shouldn't be necessary but I often run into memory issues if I'm not pedantic about this
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def extract(self, batch):
        """This is TransformerLens specific code, one could replace this using NNsight"""
        hook_pt = self.config.hook_pt
        _, cache = self.model.run_with_cache(batch, names_filter=[hook_pt], return_type="loss", stop_at_layer=self.config.layer+1)
        return cache[hook_pt]
    
    def __len__(self):
        return self.n_tokens
    
    def __getitem__(self, idx):
        """This function assumes sequential access (aka non-shuffled dataloaders). The shuffling is done automatically."""
        if idx >= self.end:
            self.collect()
            self.start = self.end
            self.end += len(self.activations)
        
        return self.activations[idx - self.start], self.tokens[idx - self.start]
            