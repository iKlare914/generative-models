# YAML 任务配置

训练配置放在 `config/train/`，采样配置放在 `config/test/`。每个原有 `.sh`
启动脚本都有一个同名 YAML；脚本只负责选择 Python 入口和配置路径。
训练 Python 入口放在项目根目录的 `train/`，采样入口放在 `sample/`。

## 启动

在项目根目录执行：

```bash
# 使用默认训练配置
bash scripts/run.sh

# 单次覆盖部分设置，不需要编辑 YAML
bash scripts/cfg-fm-run.sh --batch-size 64 --no-use-torch-compile

# 使用另一份配置
bash scripts/run.sh --config config/train/my-experiment.yaml

# 直接用 Python 入口读取 YAML
uv run --no-sync python train/diffusion_train.py --config config/train/fm-run.yaml

# 指定采样权重和输出图片
bash scripts/fm-run-sample.sh --model-path checkpoints/model.pt --output-path outputs/fm.png

# CFG 无权重运行检查（仍需加载 CLIP）
bash scripts/cfg-fm-run-sample.sh --random-init --output-path outputs/cfg-smoke.png
```

已有纯命令行方式仍然可用。所有入口的 `--help` 都列出可配置字段。

## 格式与覆盖规则

配置是平铺的 YAML 映射，字段名对应命令行参数，将 `-` 换成 `_`：

```yaml
image_size: 32
batch_size: 64
lr: 0.0001
ch_mult: [1, 2, 4]
attention_resolution: [16, 8]  # [] 关闭各层注意力，bottleneck仍保留
use_ema: true
ema_decay: 0.9999
use_torch_compile: false
use_conv: false
attn_o_proj_zeroinit: true
save_dir: ../../checkpoints/my-experiment
```

- 优先级：显式命令行参数 > YAML > Python 脚本默认值。
- boolean字段写 `true` / `false`，不要加引号；命令行用 `--no-use-ema`、
  `--no-log-samples` 等关闭配置中已开启的开关。
- 列表字段使用 YAML 列表；命令行列表覆盖整份列表。
- 未知字段、重复字段、错误类型、非法取值会报错，原有模型尺寸和训练参数校验仍然执行。
- `null` 表示尚未指定可选值；必需参数仍需在运行前补齐。
- YAML 中的 `cache_dir`、`save_dir`、`model_path`、`output_path` 等文件路径
  相对于 **YAML 所在目录**。例如 `config/train/run.yaml` 中的
  `../../checkpoints/experiment` 指向项目根目录下的 `checkpoints/experiment`。
- CLI 中的路径（包括 `--config`）仍相对于 **当前工作目录**。`.sh` 自带的
  默认配置路径为绝对路径，所以可以从其他目录启动。
- `dataset`、`text_model_name` 是字符串标识符，不按文件路径重写；如要给
  `text_model_name` 指定本地模型，建议使用绝对路径。
- CFG 采样中 `model_path` 与 `random_init: true` 互斥。命令行显式指定其中一项时，
  会覆盖 YAML 中对这组权重来源的选择。

## 参数命名

YAML 尽量使用对应构造函数的参数名，CLI 将下划线换成连字符。例如
`d_model` 对应 `--d-model`，直接传给 `DiT(d_model=...)`。

| 原字段 | 现在的字段 | 对应位置 |
| --- | --- | --- |
| `dit_hidden_size` | `d_model` | `DiT.d_model` |
| `dit_depth` | `num_layers` | `DiT.num_layers` |
| `dit_patch_size` | `patch_size` | `DiT.patch_size` |
| `dit_time_embedding` | `d_time_emb` | `DiT.d_time_emb` |
| `feature_channels` | `d_feature` | 文本维度统一名称；DiT 接收 `d_feature`，CFGUNet 接收 `feature_channel` |
| `model_channels` / `embedding_channels` | `model_channel` / `embedding_channel` | UNet 同名参数 |
| `res_blocks` / `channel_mult` / `attention_resolutions` | `resblock_num` / `ch_mult` / `attention_resolution` | UNet 同名参数 |
| `fm_step` / `timestep_spacing` | `step` / `spacing` | FM / DDIM sampler 同名参数 |
| `timesteps` | `num_timesteps` | `make_beta_schedule.num_timesteps` |
| `epochs` | `epoches` | 沿用 `TrainerConfig.epoches` 的现有拼写 |
| `vae_model_name` | `vae_model_name_or_path` | 与 `LatentDiffusion.from_pretrained_vae` 一致 |

旧字段和旧 CLI 名称仍作为别名解析；新 YAML 只写新名称。
同一字段不能同时写新旧两个名字，CLI 覆盖 YAML 时则新旧别名均有效。
`model_type` 选择模型架构，`use_latent` 选择训练数据表示；`method`、`sample` 选择算法；`vae_slicing` 对应 VAE 的方法开关，
`train_max_length` 用来区分训练 dataset 的 `max_length` 与 sampler 的 `eval_max_length`。
这些非构造函数直传项都已在 YAML 中注释。

`image_size` 在入口中始终是 RGB 图像边长；启用 latent 时，`DiT.image_size` 是从它推导的
latent `(H, W)`。`in_channel`、`grid_size_h`、`grid_size_w` 也由 VAE 和 `patch_size` 推导，
避免重复配置不一致。CFG 入口自动设置 DiT 的 `extra_feature=True`。

## 各方法适用的配置

下表列出算法专用参数；模型结构、设备、种子、路径等公共参数仍正常保留。
YAML 只填写当前方法实际使用的字段，不需要把其他方法的默认值写进去。

| 任务 | 方法 | 算法专用参数 |
| --- | --- | --- |
| 训练（含 CFG） | DDPM | `num_timesteps` |
| 训练（含 CFG） | FM | `step`、`solver`（用于训练期间记录采样结果） |
| 采样（含 CFG） | DDPM | `num_timesteps` |
| 采样（含 CFG） | DDIM | `num_timesteps`、`spacing`、`randomness` |
| 采样（含 CFG） | FM | `step`、`solver`、`randomness` |

DDIM 使用 DDPM 训练的模型，没有独立的 DDIM 训练配置。DDPM 的随机噪声由
其反向过程决定，不读取 `randomness`；该字段只控制 DDIM 的 eta 或 FM 的随机强度。

CFG 配置额外包含 `d_feature`、`text_model_name`、`eval_max_length`、
`guidance_scale`。其中训练配置另外使用 `label_drop_rate`、`train_max_length`；
采样配置另外使用 `prompt`、`random_init`、`use_cfgzero_star`、`skip_steps`。
这些字段不放入非 CFG 配置，CFG 采样开关也不放入当前训练入口的配置。

## CFG-Zero* 与独立跳步

CFG 的 DDPM、DDIM、FM 采样都支持：

```yaml
use_cfgzero_star: true
skip_steps: 5
```

`use_cfgzero_star` 控制逐张图像的投影缩放：DDPM/DDIM 作用于噪声预测，
FM 作用于速度预测。`guidance_scale: 0` 和 `1` 保持纯无条件/纯条件预测。

`skip_steps` 独立控制初始跳步，不依赖 `use_cfgzero_star`：

- `use_cfgzero_star: false`、`skip_steps: 5`：普通 CFG，仍跳过前 5 步。
- `use_cfgzero_star: true`、`skip_steps: 0`：只启用投影缩放，不跳步。
- 开启 CFG-Zero* 时必须在 YAML 或 CLI 中显式提供 `skip_steps`，否则报错。
- 关闭 CFG-Zero* 且省略 `skip_steps` 时，跳步数为 0。

跳过的步不调用模型，也不执行状态更新或添加采样噪声；初始状态保持不变，
随后在原时间表上继续采样，不重新编号时间。DDIM 按降采样后的实际步数计数，
包括额外补入的最后一个时间点。跳步数必须非负且小于实际采样总步数。

例如 `num_timesteps: 1000`、`spacing: 20` 对应 51 个 DDIM 步骤，
`skip_steps: 5` 后实际执行 46 次更新。现有 CFG DDPM/DDIM 配置保留的
`skip_steps: 5` 现在也会生效；如需完整时间表，改为 `0`。

## Latent DiT 学习实验

四个原有 Python 入口使用 `model_type` 选择架构，训练入口另用 `use_latent` 选择数据表示：

| 配置 | 模型 | 训练输入 |
| --- | --- | --- |
| 默认 `model_type: unet` | UNet / CFGUNet | RGB 图片；忽略 `use_latent`，按 false 处理 |
| `model_type: dit`、`use_latent: false` | DiT + 冻结 VAE | RGB 图片，每批在线编码 |
| `model_type: dit`、`use_latent: true` | DiT + 冻结 VAE | 本地缓存的未缩放 mean/std，不再调用 VAE encoder |

普通 UNet YAML 不填写 `use_latent`。DiT YAML 只填写 DiT 的模型参数；`d_model`、`num_layers`、
`patch_size`、`d_time_emb` 对应 DiT，UNet 则使用 `model_channel`、`embedding_channel`、`resblock_num`、
`ch_mult`、`attention_resolution` 等。两类架构分别校验、分别构造，不互相套用参数。
采样只依据 `model_type` 选择模型与解码流程，不读取训练数据，因此不使用 `use_latent`。
`image_size` 始终表示 RGB 图像尺寸；latent 的通道、下采样倍数和 scaling factor 从 VAE 配置读取。

| 任务 | 无条件配置 | CFG 配置 |
| --- | --- | --- |
| DDPM 训练 | `train/latent-ddpm.yaml` | `train/cfg-latent-ddpm.yaml` |
| FM 训练 | `train/latent-fm.yaml` | `train/cfg-latent-fm.yaml` |
| DDPM 采样 | `test/latent-ddpm.yaml` | `test/cfg-latent-ddpm.yaml` |
| DDIM 采样（用 DDPM 权重） | `test/latent-ddim.yaml` | `test/cfg-latent-ddim.yaml` |
| FM 采样 | `test/latent-fm.yaml` | `test/cfg-latent-fm.yaml` |

### 默认实验设置

- 冻结 `stabilityai/sdxl-vae`，使用 FP32；256×256 RGB 对应 4×32×32 latent。
  使用配置中的 `scaling_factor=0.13025`。启用 `vae_slicing` 降低 VAE 的峰值显存，代价是吞吐量。
- DiT：8 层、宽度 256、8 个 attention heads、latent patch size 2、time embedding 256、
  2D RoPE 和 AdaLN-Zero。每张图 256 个 image tokens；本项目实现的 DiT 参数量约 10.8M，
  加文本 cross-attention 的 CFG 版本约 15.5M，冻结 VAE 和 CLIP 不计入这两个数字。
- AdamW：`lr=1e-4`、`weight_decay=0`、EMA `0.9999`；先训练 100 epochs、每 10 epochs 保存。
  这是学习实验的起点，不是保证收敛或生成质量的训练预算。现有 DiT 没有 dropout 参数，
  DiT YAML 不填写 UNet 的 `dropout`；入口给 TrainerConfig 的占位值为 0。
- DiT 参数保持 FP32，由 trainer 在支持的 CUDA GPU 上启用 BF16 autocast；VAE 保持自己的精度策略。
  暂不开 `torch.compile`，便于调试；跑通后可使用 `--use-torch-compile`。
- DDPM：1000 步线性噪声调度。DDIM：stride 20，包含末步，共 51 步，eta 0。
  FM：uniform 连续时间训练，Euler 50 步采样，randomness 0。
- CFG：冻结 CPU CLIP `openai/clip-vit-base-patch32`，特征宽度 512，训练/采样文本长度均 32，
  空文本概率 0.1，采样 guidance 3；先关闭 CFG-Zero* 和跳步，建立基准。

### 数据与 GPU

训练 YAML 默认使用与下载工具一致的 **ImageNet-100**（`clane9/imagenet-100`），不是完整 ImageNet-1k。
这个数据源的短边已经缩到 160，256 配置会再 resize/crop；不应当把它当作原始 ImageNet-256 的复现实验。
`dataset_type: imagenet` 使用 `image` / `label` 字段及现有 resize cache，CFG 额外需要
`ClassLabel.names` 中的人类可读类别名称。只使用 `dataset_split: train`，避免把 validation 混入训练。
训练日志的固定 prompts 从当前 ImageNet 类别列表中取。所有采样 YAML 的 `prompt` 默认是空字符串；
采样时空字符串直接交给 text encoder 作为无条件文本，12 张图全部使用空条件，有效 guidance scale 为 0。
需要条件生成时通过 `--prompt "a photo of a pineapple"` 指定，12 张图均使用该文本和配置中的 guidance scale。
训练 YAML 无需 `prompt`；训练入口即使收到此字段或 `--prompt` 也会忽略，训练条件及日志示例由 dataset 提供。
更换 ImageNet 数据源时设置 `dataset`，并确认上述字段约定。

RTX PRO 6000 Blackwell 为 96GB，RTX 5090 为 32GB。初始建议分别为 batch 128 和 32，
其余模型/算法参数保持一致；未在这两张卡上实测峰值显存。batch 改变也会改变每 epoch 的更新次数
及 EMA 对应的样本跨度，因此适合功能测试，但不是严格等价的训练对照。

本机用已有 CIFAR 验证时，覆盖 `dataset_type`、`dataset`、`cache_dir` 和 split，并加 `--no-use-latent` 使用在线编码（仍为 DiT）。
下面的 CPU smoke 命令将 CIFAR 缩放到 32，以缩短检查时间；它仅验证流程，不能用于判断 ImageNet 生成质量：

```bash
# 安装依赖应在已有环境之外按需执行；不需要另建训练/采样入口。
uv sync --extra latent-diffusion --extra clip-cfg

# 先生成 ImageNet-100 的 latent 缓存（只需一次；默认优先 CUDA，--device cpu 可显式覆盖）
uv run --no-sync python utils/download_and_process_imagenet.py \
  --repo-name clane9/imagenet-100 --split train --use-latent --image-size 256 \
  --vae-model-name-or-path stabilityai/sdxl-vae --batch-size 8

# PRO 6000：ImageNet-100 / CFG / FM，直接读取上述缓存
uv run --no-sync python train/cfg_diffusion_train.py --config config/train/cfg-latent-fm.yaml

# 5090：同一个网络，先降低 batch size
uv run --no-sync python train/cfg_diffusion_train.py --config config/train/cfg-latent-fm.yaml --batch-size 32

# 复用默认位置的本地 CIFAR 缓存，禁止数据和模型联网下载。
# VAE 不在本地时，需要将 --vae-model-name-or-path 指向已有的 VAE 本地目录。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run --no-sync python train/diffusion_train.py \
  --config config/train/latent-fm.yaml --no-use-latent --dataset-type cifar10 --dataset uoft-cs/cifar10 \
  --dataset-split 'train[:8]' --cache-dir "$HOME/.cache/huggingface/datasets" \
  --image-size 32 --batch-size 2 --epoches 1 --save-interval-epoch 1 \
  --device cpu --num-workers 0 --step 0.5 --wandb-mode disabled \
  --save-dir outputs/latent-cifar-smoke

# ImageNet 实验采样；需要选择实际 checkpoint 路径。
uv run --no-sync python sample/cfg_diffusion_sample.py --config config/test/cfg-latent-fm.yaml \
  --model-path checkpoints/imagenet100-cfg-latent-fm/实际文件名_checkpoint.pt
```

采样时 VAE、图像尺寸和 DiT 结构必须与训练一致。CIFAR smoke 的权重应覆盖 `--image-size 32`，
并传入 smoke 使用的同一个 VAE。采样入口优先使用 DiT-only EMA，同时从完整 checkpoint 恢复 VAE；
没有 EMA 时恢复完整模型，也兼容 `torch.compile` 添加的 DiT 权重前缀。
`sample_posterior` 同时控制在线 VAE 编码和缓存 posterior 的采样：true 使用 `mean + std * noise`，
false 使用 mean。两条路径都由 trainer/模型约定完成一次 scaling factor 变换。

预处理脚本默认仍使用原来的 `split=all`；上面的命令显式选择 `--split train`，与训练 YAML 一致。
`--use-latent` 隐含按 `--image-size` resize/crop，之后调用 `to_latent`；默认保存到
`datasets/imagenet-100-256x256-train-latent`，包含原始 mean/std 和标签元数据。
预处理的 `--save-dir` 与训练的 `latent_cache_dir` 必须指向同一缓存根目录，二者都不是 checkpoint 目录。
训练入口不会自动下载源图片或创建缺失的 latent 缓存；缺失时会提示先运行预处理脚本。
CFG 缓存 loader 返回 `((mean, std), (context, attention_mask))`，用 `PROMPT_TEMPLATES` 缓存类别文本条件。
不同 VAE 使用不同缓存目录，且预处理、训练和采样使用同一个 VAE。

### 配置依据

- [官方 DiT 模型配置](https://github.com/facebookresearch/DiT/blob/main/models.py)：DiT-S/2 为 12 层、384 宽、6 heads。
  本配置主动缩小到 8 层、256 宽，使用项目现有的 RoPE/cross-attention 实现，并非原论文的原样复现。
- [官方 DiT 训练代码](https://github.com/facebookresearch/DiT/blob/main/train.py)：AdamW 1e-4、weight decay 0、EMA 0.9999；
  本项目 FM 配置复用这些值作为实验起点，不声称是官方 FM 最优参数。
- [SDXL VAE 配置](https://huggingface.co/stabilityai/sdxl-vae/blob/main/config.json)、
  [ImageNet-100 数据卡](https://huggingface.co/datasets/clane9/imagenet-100)。
- [RTX PRO 6000 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)、
  [RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)。
