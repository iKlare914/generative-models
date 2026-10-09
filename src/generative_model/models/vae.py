import torch as th
from torch import nn

from .attention import AttentionBlock
from .unet import DownSampleBlock, UpSampleBlock
from .utils import zero_init


class ResidualBlock(nn.Module):
    """
    Residual connection for UNet
    """
    def __init__(self, in_channel, out_channel, dropout, is_upsample=False, is_downsample=False, use_conv=False):
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

        if in_channel == out_channel:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x: th.Tensor):
        """
        Args:
            x: Tensor [B, C, H, W]
        Returns:
            result: Tensor [B, C, H, W]
        """
        h = self.in_layers(x)
        h = self.h_upd(h)
        h = self.conv(h)
        x = self.x_upd(x)

        h = self.out_layers(h)
        return self.skip_connection(x) + h


class Encoder(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            model_channel,
            z_channel,
            resblock_num,
            dropout=0,
            ch_mult=(1, 2, 3, 4),
            attention_resolution=(16, 8, 4),
            num_heads=1,
            attn_o_proj_zeroinit=False,
            image_size=32,
            use_conv=False
    ):
        super().__init__()
        self.image_size = image_size
        self.attention_resolution = tuple(attention_resolution)
        self.in_channel = in_channel
        self.model_channel = model_channel
        self.z_channel = z_channel
        self.out_channel = out_channel
        self.resblock_num = resblock_num
        self.dropout = dropout
        self.num_heads = num_heads
        self.attn_o_proj_zeroinit = attn_o_proj_zeroinit
        self.use_conv = use_conv
        downsample_scale_factor = 1

        # Encoder Part
        self.encoder = nn.ModuleList([nn.Conv2d(in_channel, model_channel, 3, padding=1)]) # From RGB 3 channel to model channel
        cur_ch = model_channel
        encoder_chs = [cur_ch]
        for level, mult in enumerate(ch_mult):
            for _ in range(resblock_num):
                layers = []
                resblock = ResidualBlock(
                    cur_ch,
                    int(model_channel * mult),
                    dropout,
                    is_upsample=False,
                    is_downsample=False,
                    use_conv=use_conv
                )
                cur_ch = int(model_channel * mult)
                layers.append(resblock)
                if image_size // downsample_scale_factor in attention_resolution:
                    attention_block = AttentionBlock(
                        cur_ch,
                        num_heads,
                        attn_o_proj_zeroinit
                    )
                    layers.append(attention_block)
                self.encoder.append(nn.Sequential(*layers))
                encoder_chs.append(cur_ch)
            if level != len(ch_mult) - 1:
                self.encoder.append(
                    nn.Sequential(
                        ResidualBlock(
                            cur_ch,
                            cur_ch,
                            dropout,
                            is_downsample=True,
                            use_conv=use_conv
                        )
                    )
                )
                downsample_scale_factor *= 2
                encoder_chs.append(cur_ch)

        self.out_layer = nn.Sequential(
            ResidualBlock(
                cur_ch,
                cur_ch,
                dropout,
                use_conv=use_conv
            ),
            AttentionBlock(
                cur_ch,
                num_heads,
                attn_o_proj_zeroinit
            ),
            ResidualBlock(
                cur_ch,
                cur_ch,
                dropout,
                use_conv=use_conv
            ),
            nn.GroupNorm(32, cur_ch),
            nn.SiLU(),
            nn.Conv2d(cur_ch, 2 * z_channel, kernel_size=3, padding=1) # mean channel and log std channel
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        for layer in self.encoder:
            x = layer(x)
        return self.out_layer(x)


class Decoder(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            model_channel,
            z_channel,
            resblock_num,
            dropout=0,
            ch_mult=(1, 2, 3, 4),
            attention_resolution=(16, 8, 4),
            num_heads=1,
            attn_o_proj_zeroinit=False,
            image_size=32,
            use_conv=False
    ):
        super().__init__()
        self.image_size = image_size
        self.attention_resolution = tuple(attention_resolution)
        self.in_channel = in_channel
        self.model_channel = model_channel
        self.z_channel = z_channel
        self.out_channel = out_channel
        self.resblock_num = resblock_num
        self.dropout = dropout
        self.num_heads = num_heads
        self.use_conv = use_conv
        self.attn_o_proj_zeroinit = attn_o_proj_zeroinit
        downsample_scale_factor = 2 ** (len(ch_mult) - 1)
        cur_ch = model_channel * ch_mult[-1]

        self.in_layer = nn.Sequential(
            nn.Conv2d(z_channel, cur_ch, kernel_size=3, padding=1),
            ResidualBlock(
                cur_ch,
                cur_ch,
                dropout,
                use_conv=use_conv
            ),
            AttentionBlock(
                cur_ch,
                num_heads,
                attn_o_proj_zeroinit
            ),
            ResidualBlock(
                cur_ch,
                cur_ch,
                dropout,
                use_conv=use_conv
            )
        )

        # Decoder Part
        self.decoder = nn.ModuleList([self.in_layer])
        for level, mult in list(enumerate(ch_mult))[::-1]:
            for i in range(resblock_num + 1):
                layers = [
                    ResidualBlock(
                        cur_ch,
                        int(model_channel * mult),
                        dropout,
                        use_conv=use_conv
                    )
                ]
                cur_ch = int(model_channel * mult)
                if image_size // downsample_scale_factor in attention_resolution:
                    layers.append(
                        AttentionBlock(
                            cur_ch,
                            num_heads,
                            attn_o_proj_zeroinit
                        )
                    )
                if level and i == resblock_num:
                    layers.append(
                        ResidualBlock(
                            cur_ch,
                            cur_ch,
                            dropout,
                            is_upsample=True,
                            use_conv=use_conv
                        )
                    )
                    downsample_scale_factor //= 2
                self.decoder.append(nn.Sequential(*layers))


    def forward(self, x: th.Tensor) -> th.Tensor:
        for layer in self.decoder:
            x = layer(x)
        return x


class VariationalAutoEncoder(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            model_channel,
            z_channel,
            emb_channel, # channel of posterior sample, can be differennt from z channel
            resblock_num,
            dropout=0,
            ch_mult=(1, 2, 3, 4),
            attention_resolution=(16, 8, 4),
            num_heads=1,
            attn_o_proj_zeroinit=False,
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
        self.z_channel = z_channel
        self.out_channel = out_channel
        self.resblock_num = resblock_num
        self.dropout = dropout
        self.num_heads = num_heads
        self.use_conv = use_conv

        # Encoder Part
        self.encoder = Encoder(
            in_channel,
            out_channel,
            model_channel,
            z_channel,
            resblock_num,
            dropout,
            ch_mult,
            attention_resolution,
            num_heads,
            attn_o_proj_zeroinit,
            image_size,
            use_conv
        )

        self.posterior_conv = nn.Conv2d(2 * z_channel, 2 * emb_channel, kernel_size=1)

        # Decoder Part
        self.decoder = Decoder(
            in_channel,
            out_channel,
            model_channel,
            z_channel,
            resblock_num,
            dropout,
            ch_mult,
            attention_resolution,
            num_heads,
            attn_o_proj_zeroinit,
            image_size,
            use_conv
        )

        self.post_posterior_conv = nn.Conv2d(emb_channel, z_channel, kernel_size=1)

        # Convert feature to result
        self.out = nn.Sequential(
            nn.GroupNorm(32, int(model_channel * ch_mult[0])),
            nn.SiLU(),
            nn.Conv2d(int(model_channel * ch_mult[0]), out_channel, 3, padding=1)
        )

    def encode(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Encode larger image into smaller latent representation
        """
        x = self.encoder(x)
        mean, log_std = self.posterior_conv(x).chunk(2, dim=1)
        log_std = th.clamp(log_std, -15.0, 10.0) # ref: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/distributions/distributions.py , but we predict log std instead of log variance
        z = mean + th.randn_like(mean) * th.exp(log_std)
        return z, mean, log_std

    def decode(self, x: th.Tensor) -> th.Tensor:
        """
        Decode original image from latent representation
        """
        x = self.post_posterior_conv(x)
        x = self.decoder(x)
        return self.out(x)

    def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        UNet forward
        Args:
            x: Image tensor [B, C, H, W]
        Returns:
            out: Reconstructed image tensor [B, C, H, W]
            mean: mean of posterior, tensor with shape [B, emb_channel, H', W']
            log_std: log std of posterior, tensor with shape [B, emb_channel, H', W']
        """
        z, mean, log_std = self.encode(x)
        out = self.decode(z)
        return out, mean, log_std
