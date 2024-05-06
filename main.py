# %%
%load_ext autoreload
%autoreload 2

from transformer_lens import HookedTransformer
import plotly.express as px
import torch
from einops import *

from . import get_splits, Config, Sampler, AnthropicSAE, VanillaSAE, GatedSAE, RainbowSAE

# %%
model = HookedTransformer.from_pretrained("gelu-1l").cuda()
train, validation = get_splits()

# %%

# This equates to about 65M tokens
config = Config(n_buffers=500, expansion=4, buffer_size=2**17, sparsities=(0.01, 0.02, 0.04, 0.07, 0.14, 0.27, 0.52, 1.00))
sampler = Sampler(config, train, model)
sae = RainbowSAE(config, model).cuda()

# %%
torch.backends.cudnn.benchmark = True
sae.train(sampler, model, validation, log=True)

# %%