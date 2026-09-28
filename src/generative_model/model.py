import torch as th
from torch.nn import functional as F
from torch import nn
import tqdm.auto as tqdm
from abc import abstractmethod
import math

def zero_init(module: nn.Module):
    """
    Initialize all parameters of a module with zero
    """
    for p in module.parameters():
        p = p.detach().zero_()
    return module

def position_embedding(x: th.Tensor, emb_dim: int):
    """
    Apply position embedding for input
    Args:
        x: Tensor [B]
        emb_dim: Dimension of embedding
    Returns:
        emb: Tensor [B, emb_dim]
    """
    if len(x.shape) != 1:
        raise ValueError("x must be 1D tensor")
    if emb_dim <= 0 or not isinstance(emb_dim, int) or emb_dim % 2:
        raise ValueError("emb_dim must be a positive even integer")
    d = emb_dim // 2
    scale = math.log(10000) / d
    scale = th.exp(th.arange(d, dtype=th.float32, device=x.device) * -scale)
    emb = th.outer(x.float(), scale)
    emb = th.cat(
        [th.sin(emb), th.cos(emb)],
        dim=-1
    )
    return emb

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, use_conv=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.use_conv = use_conv

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, use_conv=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.use_conv = use_conv
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2) 
    def forward(self, x):
        x = self.avg_pool(x) 
        if self.use_conv:
            x = self.conv(x)
        return x

class TimeEmbeddedBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply time embedding to input x
        """

class ResidualBlock(TimeEmbeddedBlock):
    """
    Residual connection for UNet
    """
    def __init__(self, in_channel, out_channel, emb_channel, dropout, is_upsample=False, is_downsample=False, use_conv=False):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            nn.SiLU()
        )
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        if is_upsample:
            self.x_upd = UpSampleBlock(out_channel, out_channel, 2, use_conv)
            self.h_upd = UpSampleBlock(out_channel, out_channel, 2, use_conv)
        elif is_downsample:
            self.x_upd = DownSampleBlock(out_channel, out_channel, 2, use_conv)
            self.h_upd = DownSampleBlock(out_channel, out_channel, 2, use_conv)
        else:
            self.x_upd = nn.Identity()
            self.h_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channel),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_init(nn.Conv2d(out_channel, out_channel, 3, padding=1))
        )

        # get independent representation of time embedding for every block
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channel, out_channel)
        )
        if in_channel == out_channel:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x: th.Tensor, emb: th.Tensor):
        """
        Args:
            x: Tensor [B, C, H, W]
            emb: Tensor [N, C]
        Returns:
            result: Tensor [B, C, H, W]
        """
        h = self.in_layers(x)
        h = self.h_upd(h)
        h = self.conv(h)
        x = self.x_upd(x)

        emb = self.emb_layer(emb)
        emb = emb.unsqueeze(-1).unsqueeze(-1)

        h = self.out_layers(emb + h)
        return self.skip_connection(x) + h

class TimeSequentialBlock(nn.Sequential, TimeEmbeddedBlock):
    """
    A sequential module that dispatch appropriate input for modules
    """
    def forward(self, x, emb):
        """
        Args:
            x: Tensor [B, C, H, W]
            emb: Tensor [N, C]
        """

        for layer in self:
            if isinstance(layer, TimeEmbeddedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class QKVMHAttention(nn.Module):
    def __init__(self, num_heads = 1):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv: th.Tensor):
        """
        Args:
            qkv: Tensor [B, C, (H*W)]
        Returns:
            attention: Tensor [B, C, (H*W)]
        """
        b, c, l = qkv.shape
        q, k, v = qkv.chunk(3, dim=1)
        if (c // 3) % self.num_heads != 0:
            raise ValueError(f"Channel({c // 3}) cannot be divided by num heads({self.num_heads})")
        head_channel = (c // 3) // self.num_heads
        scale = 1 / math.sqrt(math.sqrt(head_channel))
        q = q.reshape(b * self.num_heads, head_channel, l) * scale
        k = k.reshape(b * self.num_heads, head_channel, l) * scale
        v = v.reshape(b * self.num_heads, head_channel, l)
        weight =  th.softmax(th.einsum("bcl,bct->blt", q, k), dim=-1)
        attention = th.einsum(
            "blt,bct->bcl",
            weight,
            v
        )
        return attention.reshape(b, c // 3, l)

class AttentionBlock(nn.Module):
    def __init__(self, channel, num_heads = 1):
        super().__init__()
        self.channel = channel
        self.num_heads = num_heads
        self.qkv_proj = nn.Conv1d(channel, 3 * channel, 1)
        self.o_proj = zero_init(nn.Conv1d(channel, channel, 1))
        self.norm = nn.GroupNorm(32, channel)
        self.attention = QKVMHAttention(num_heads)

    def forward(self, x: th.Tensor):
        """
        Args:
            x: Tensor [B, C, H, W]
        Returns: 
            result: Tensor [B, C, H, W]
        """
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv_proj(self.norm(x))
        a = self.attention(qkv)
        out = self.o_proj(a) + x
        return out.reshape(b, c, h, w)

class UNet(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            model_channel,
            embedding_channel,
            resblock_num,
            dropout=0,
            ch_mult = (1, 2, 3, 4),
            attention_resolution = (16, 8, 4),
            num_heads = 1,
            num_class = None,
            dtype=th.float32,
            image_size=32,
            use_conv=False
    ):
        super().__init__()
        if not ch_mult:
            raise ValueError("ch_mult must contain at least one level")
        max_scale = 2 ** (len(ch_mult) - 1)
        if image_size <= 0 or image_size % max_scale:
            raise ValueError("image_size must be positive and divisible by the maximum downsampling factor")
        valid_resolutions = {image_size // (2 ** level) for level in range(len(ch_mult))}
        if not set(attention_resolution).issubset(valid_resolutions):
            raise ValueError(f"attention_resolution must contain feature-map sizes from {sorted(valid_resolutions)}")
        self.image_size = image_size
        self.attention_resolution = tuple(attention_resolution)
        self.in_channel = in_channel
        self.model_channel = model_channel
        self.embedding_channel = embedding_channel
        self.out_channel = out_channel
        self.resblock_num = resblock_num
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_class = num_class
        self.dtype = dtype
        self.use_conv = use_conv
        downsample_scale_factor = 1
        if num_class:
            self.class_emb_layer = nn.Embedding(num_class, embedding_channel)

        # Encoder Part
        self.encoder = nn.ModuleList([TimeSequentialBlock(nn.Conv2d(in_channel, model_channel, 3, padding=1))]) # From RGB 3 channel to model channel
        cur_ch = model_channel
        encoder_chs = [cur_ch]
        for level, mult in enumerate(ch_mult):
            for _ in range(resblock_num):
                layers = []
                resblock = ResidualBlock(
                    cur_ch,
                    int(model_channel * mult),
                    embedding_channel,
                    dropout,
                    is_upsample=False,
                    is_downsample=False
                )
                cur_ch = int(model_channel * mult)
                layers.append(resblock)
                if image_size // downsample_scale_factor in attention_resolution:
                    attention_block = AttentionBlock(
                        cur_ch,
                        num_heads
                    )
                    layers.append(attention_block)
                self.encoder.append(TimeSequentialBlock(*layers))
                encoder_chs.append(cur_ch)
            if level != len(ch_mult) - 1:
                self.encoder.append(
                    TimeSequentialBlock(
                        ResidualBlock(
                            cur_ch,
                            cur_ch,
                            embedding_channel,
                            dropout,
                            is_downsample=True   
                        )
                    )
                )
                downsample_scale_factor *= 2
                encoder_chs.append(cur_ch)

        # Bottleneck Part
        self.bottleneck = TimeSequentialBlock(
            ResidualBlock(
                cur_ch,
                cur_ch,
                embedding_channel,
                dropout
            ),
            AttentionBlock(
                cur_ch,
                num_heads
            ),
            ResidualBlock(
                cur_ch,
                cur_ch,
                embedding_channel,
                dropout
            )
        )

        # Decoder Part
        self.decoder = nn.ModuleList([])
        for level, mult in list(enumerate(ch_mult))[::-1]:
            for i in range(resblock_num + 1):
                encoder_out_ch = encoder_chs.pop()
                layers = [
                    ResidualBlock(
                        encoder_out_ch + cur_ch,
                        int(model_channel * mult),
                        embedding_channel,
                        dropout    
                    )
                ]
                cur_ch = int(model_channel * mult)
                if image_size // downsample_scale_factor in attention_resolution:
                    layers.append(
                        AttentionBlock(
                            cur_ch,
                            num_heads
                        )
                    )
                if level and i == resblock_num:
                    layers.append(
                        ResidualBlock(
                            cur_ch,
                            cur_ch,
                            embedding_channel,
                            dropout,
                            is_upsample=True
                        )
                    )
                    downsample_scale_factor //= 2
                self.decoder.append(TimeSequentialBlock(*layers))

        # Convert feature to result
        self.out = nn.Sequential(
            nn.GroupNorm(32, cur_ch),
            nn.SiLU(),
            zero_init(nn.Conv2d(cur_ch, out_channel, 3, padding=1))
        )

    def forward(self, x, t, y=None):
        """
        UNet forward
        Args:
            x: Image tensor [B, C, H, W]
            t: Time step tensor [B]
            y: Class label tensor [B]
        Returns:
            out: Predicted standard gaussian noise tensor [B, C, H, W]
        """
        emb = position_embedding(t, self.embedding_channel)
        if self.num_class is not None:
            emb += self.class_emb_layer(y)
        sc = []
        h = x.type(self.dtype)
        for layer in self.encoder:
            h = layer(h, emb)
            sc.append(h)
        h = self.bottleneck(h, emb)
        for layer in self.decoder:
            h = th.cat(
                [sc.pop(), h],
                dim=1
            )
            h = layer(h, emb)
        out = self.out(h)
        return out