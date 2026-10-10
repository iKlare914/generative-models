import torch as th
from torch import nn
import math
from generative_model.models.attention import QKVMHACrossAttention, QKVMHAttention
from generative_model.models.norms import AdaptiveLayerNorm
from generative_model.models.utils import ada_modulation

# ref: https://github.com/facebookresearch/DiT/blob/main/models.py

# TODO
# 1. DiTBlock
# 2. DiT
# 3. Patchify -- UnPatchify 
# 4. 2d sinusoidal positional embedding
# 5. 2d-Rope
# 6. AdaLN condition injection
# 7. CrossAttention
# 8. weight init


