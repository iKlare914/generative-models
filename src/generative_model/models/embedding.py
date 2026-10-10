import torch as th
from torch import nn
from generative_model.models.utils import position_embedding
import math

class TimeStepEmbedder(nn.Module):
    def __init__(self, freq_dim, hidden_dim):
        super().__init__()
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_dim
        self.out = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: th.Tensor):
        """
        Add nonlinearity for timestep embedding
        Args:
            x: Tensor [B]
        Returns:
            Tensor [B, hidden_dim]
        """
        x = position_embedding(x, self.freq_dim)
        return self.out(x)

class RoPE2D(nn.Module):
    def __init__(self, grid_size_h, grid_size_w, theta=10000.0):
        super().__init__()
        self.base_theta = theta
        self.grid_size_h = grid_size_h
        self.grid_size_w = grid_size_w
        gh, gw = th.meshgrid(th.arange(grid_size_h), th.arange(grid_size_w), indexing='ij') # on cpu by default
        self.register_buffer('gh', gh.flatten(), persistent=False)
        self.register_buffer('gw', gw.flatten(), persistent=False)

    def _compute_freq_comp(self, pos: th.Tensor, feature_dim: int) -> tuple[th.Tensor, th.Tensor]:
        """
        Calculate sin theta and cos theta
        Args:
            pos: position of patch [grid_size_h * grid_size_w]
            feature_dim: dimension of feature, must be even
        Return:
            sin_comp: sin component of shape [1, 1, grid_size_h * grid_size_w, feature_dim]
            cos_comp: cos component of shape [1, 1, grid_size_h * grid_size_w, feature_dim]
        """
        if feature_dim % 2 != 0:
            raise ValueError("feature dim must be even")
        d = feature_dim // 2
        scale = math.log(self.base_theta) / d
        freq = th.exp(th.arange(d, device=pos.device, dtype=th.float32) * -scale) # torch.arange on cpu by default
        freq = th.outer(pos.float(), freq)
        sin_freq = th.sin(freq)
        cos_freq = th.cos(freq)
        return (
            th.concat((sin_freq, sin_freq), dim=-1).unsqueeze_(0).unsqueeze_(0),
            th.concat((cos_freq, cos_freq), dim=-1).unsqueeze_(0).unsqueeze_(0)
         ) # repeat sin/cos freq

    def _rotate_feature(self, tokens: th.Tensor) -> th.Tensor:
        """
        Rotate feature for elementwise multiplication in future operation
        Args:
            tokens: Tensor [B, H, L, D]
        Returns:
            tokens: Tensor [B, H, L, D]
        """
        t1, t2 = tokens.chunk(2, dim=-1)
        return th.concat((-t2, t1), dim=-1)

    def _apply_1d_rope(self, tokens: th.Tensor, sin_comp: th.Tensor, cos_comp: th.Tensor) -> th.Tensor:
        """
        Apply 1d rope
        Args:
            tokens: Tensor [B, H, L, D // 2]
            sin_comp: Tensor [1, 1, L, D // 2]
            cos_comp: Tensor [1, 1, L, D // 2]

        Returns:
            tokens: Tensor [B, H, L, D]
        """
        return tokens * cos_comp + self._rotate_feature(tokens) * sin_comp

    def forward(self, tokens: th.Tensor) -> th.Tensor:
        """
        Args:
            tokens: Tensor [B, H, L, D]
        """
        if tokens.shape[-1] % 4 != 0:
            raise ValueError("hidden dim must be divisible by 4")
        if tokens.shape[2] != self.grid_size_h * self.grid_size_w:
            raise ValueError("token count must match grid size")
        hd = tokens.shape[-1]
        device = tokens.device
        dtype = tokens.dtype

        gh_sin_comp, gh_cos_comp = self._compute_freq_comp(self.gh.to(tokens.device), hd // 2)
        gw_sin_comp, gw_cos_comp = self._compute_freq_comp(self.gw.to(tokens.device), hd // 2)

        gh_tokens, gw_tokens = tokens.chunk(2, dim=-1)
        gh_rope = self._apply_1d_rope(gh_tokens, gh_sin_comp, gh_cos_comp)
        gw_rope = self._apply_1d_rope(gw_tokens, gw_sin_comp, gw_cos_comp)
        
        return th.concat((gh_rope, gw_rope), dim=-1).type(dtype)



def getSinusoidalEmbedding2D(freq_dim: int, grid_size_h: int, grid_size_w: int) -> th.Tensor:
    """
        Apply axial sinusoidal embedding for 2D
        Args:
            freq_dim: Dimension of embedding
            grid_size_h: Number of patches along the height
            grid_size_w: Number of patches along the width
        Return:
            emb: [1, grid_size_h * grid_size_w, freq_dim]
    """
    if not isinstance(freq_dim, int) or freq_dim <= 0 or freq_dim % 4 != 0:
        raise ValueError(f"freq_dim must be a positive integer divisible by 4, but got {freq_dim}")
    gh, gw = th.meshgrid(th.arange(grid_size_h), th.arange(grid_size_w), indexing="ij") # [grid_size_h, grid_size_w]
    gh_emb = position_embedding(gh.reshape(-1), freq_dim // 2) # [grid_size_h * grid_size_w, freq_dim // 2]
    gw_emb = position_embedding(gw.reshape(-1), freq_dim // 2)
    return th.cat([gw_emb, gh_emb], dim=-1).unsqueeze(0) # [1, grid_size_h * grid_size_w, freq_dim]


    
