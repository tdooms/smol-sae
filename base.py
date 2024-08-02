from torch import nn
import torch
from transformer_lens import utils
from einops import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from transformers import PreTrainedModel, PretrainedConfig
import wandb
from tqdm import tqdm
import gc

class Config(PretrainedConfig):
    def __init__(
        self,
        point : str | None = None,
        layer : int | None = None,
        d_in: int | None = None,
        n_ctx: int = 256,           # context window size to sample
        n_tokens: int = 2**24,      # ~16M tokens
        in_batch: int = 32,         # batch size for the transformer
        out_batch: int = 4096,      # batch size for the SAE
        buffer_size: int = 2**17,   # ~250k tokens
        expansion: int = 16,        # SAE expansion factor
        lr: float = 1e-4,
        sparsity: float | int = 1.0,
        validation_interval: int = 1000,
        dead_thresh: int = 5,
        device = "cuda",
    ):
        assert point is not None and layer is not None, "Must specify where to hook the SAE"
        assert d_in is not None, "Must specify the input dimension SAE"
        
        self.point = point
        self.layer = layer
        
        self.d_in = d_in
        self.n_ctx = n_ctx
        
        self.buffer_size = buffer_size
        self.n_tokens = n_tokens
        self.in_batch = in_batch
        self.out_batch = out_batch
        
        self.expansion = expansion
        self.lr = lr
        self.sparsity = sparsity
        
        self.validation_interval = validation_interval
        self.inactive_thresh = dead_thresh
        
        self.hook_pt = utils.get_act_name(point, layer)
        self.device = device
    

class BaseSAE(PreTrainedModel):
    """
    Base class for all Sparse Auto Encoders.
    Provides a common interface for training and evaluation.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        
        # Set important parameters
        self.n_ctx = config.n_ctx
        self.d_in = config.d_in
        self.d_hidden = self.config.expansion * self.d_in
        
        # Initialize the decoder and normalize the output dim
        self.decoder = nn.Linear(self.d_hidden, self.d_in, bias=True)
        self.decoder.weight.data /= torch.norm(self.decoder.weight.data, dim=-2, keepdim=True)
        
        # Initialize the encoder to the transpose
        self.encoder = nn.Linear(self.d_in, self.d_hidden, bias=True)
        self.encoder.weight.data = self.decoder.weight.data.T.clone()
        
        # Initialize add biases to zero
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
    
    def forward(self, x, y=None):
        return self.decode(self.encode(x), y)
    
    def decode(self, x, y=None):
        return self.decoder(x)
    
    def encode(self, x):
        raise NotImplementedError
    
    @classmethod
    def from_pretrained(cls, path, device="cuda", **kwargs):
        config = Config.from_pretrained(path)
        return super(BaseSAE, BaseSAE).from_pretrained(path, config=config, device_map=device, **kwargs)
    
    @classmethod
    def from_config(cls, *args, **kwargs):
        return BaseSAE(Config(*args, **kwargs))
    
    def loss(self, x, x_hid, x_hat, *args):
        raise NotImplementedError
    
    def metrics(self, x, x_hid, x_hat, *args):
        self.inactive[x_hid.sum(dim=0) > 0] = 0
        self.inactive += 1
        
        mse = (x - x_hat).pow(2).sum(-1)
        
        metrics = {
            "step": self.step,
            "train/dead": (self.inactive > self.config.inactive_thresh).float().mean().item(),
            "train/mse": mse.mean(),
            "train/nmse": (mse / x.pow(2).sum(-1)).mean(),
            "train/l1": x_hid.sum(dim=-1).mean().item(),
            "train/l0": (x_hid > 0).float().sum(dim=-1).mean().item()
        }
        
        return metrics
    
    def train(self, train, model, validation, log: str | None = "sae"):
        if log is not None: wandb.init(project=log)

        # Set (global) training variables
        self.inactive = torch.zeros(self.d_hidden)
        self.step = 0
        self.steps = self.config.n_tokens // self.config.out_batch

        # scheduler = LambdaLR(self.optimizer, lr_lambda=lambda t: min(5*(1 - t/self.steps), 1.0))
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.steps, eta_min=1e-5)

        # Data loader & progress bar
        loader = DataLoader(train, batch_size=self.config.out_batch, drop_last=True)
        pbar = tqdm(loader, total=self.steps)
        
        # Main training loop
        for x, y in pbar:
            x_hid = self.encode(x)
            x_hat = self.decode(x_hid, y)

            loss = self.loss(x, x_hid, x_hat)
            metrics = self.metrics(x, x_hid, x_hat)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            scheduler.step()

            # Perform a validation step to check the final CE once in a while
            if self.step % self.config.validation_interval == 0:
                clean_loss, corrupt_loss, loss = self.patch_loss(model, validation)
                metrics["val/ce_added"] = (loss.item() - clean_loss) / clean_loss
                metrics["val/ce_recovered"] = 1 - (loss  - clean_loss) / (corrupt_loss - clean_loss)

            pbar.set_description(f"NMSE: {metrics['train/nmse']:.4f}")
            
            if log is not None: wandb.log(metrics)
            self.step += 1
        
        if log is not None: wandb.finish()

    @torch.inference_mode()
    def patch_loss(self, model, validation):
        validation = validation["input_ids"][:, :self.config.n_ctx].to(self.config.device)
        
        hook_pt = self.config.hook_pt
        clean_loss, cache = model.run_with_cache(validation, return_type="loss", names_filter=[hook_pt])
        
        corrupt_hook = lambda act, hook: torch.zeros_like(act)
        corrupt_loss = model.run_with_hooks(validation, return_type="loss", fwd_hooks=[(hook_pt, corrupt_hook)])
        
        x_hat = self.forward(cache[hook_pt], validation)
        patch_hook = lambda act, hook: x_hat
        
        loss = model.run_with_hooks(validation, return_type="loss", fwd_hooks=[(hook_pt, patch_hook)])
        return clean_loss, corrupt_loss, loss
    