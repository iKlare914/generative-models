import torch as th
from torch import nn
from generative_model.models.utils import position_embedding

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

class Rope2D(nn.Module):
    pass

def getSinusoidalEmbedding2D(freq_dim: int, grid_size: int) -> th.Tensor:
    """
        Apply axial sinusoidal embedding for 2D
        Args:
            freq_dim: Dimension of embedding
            grid_size: Number of patches along each side of a square grid
        Return:
            emb: [1, grid_size ** 2, freq_dim]
    """
    if not isinstance(freq_dim, int) or freq_dim <= 0 or freq_dim % 4 != 0:
        raise ValueError(f"freq_dim must be a positive integer divisible by 4, but got {freq_dim}")
    gh, gw = th.meshgrid(th.arange(grid_size), th.arange(grid_size), indexing="ij") # [grid_size, grid_size]
    gh_emb = position_embedding(gh.reshape(-1), freq_dim // 2) # [grid_size ** 2, freq_dim // 2]
    gw_emb = position_embedding(gw.reshape(-1), freq_dim // 2)
    return th.cat([gw_emb, gh_emb], dim=-1).unsqueeze(0) # [1, grid_size ** 2, freq_dim]


    
