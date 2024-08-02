# %%
%load_ext autoreload
%autoreload 2

from transformer_lens import HookedTransformer
import torch

from base import Config
from utils import get_dataset_splits, TokenDataset
from normed import NormedSAE
from vanilla import VanillaSAE
from topk import TopkSAE
from tokenized import TokenizedSAE

# %%
model = HookedTransformer.from_pretrained("gpt2").cuda()
train, validation = get_dataset_splits(model.tokenizer, n_validation=16)
# %%

config = Config(point="resid_pre", layer=8, d_in=768, n_tokens=2**24, expansion=8, buffer_size=2**19, sparsity=30)
ds = TokenDataset(config, model, train)

# sae = TopkSAE(config).cuda()
sae = TokenizedSAE(config, model).cuda()
# %%
torch.backends.cudnn.benchmark = True
sae.train(ds, model, validation, log="sae")
# %%