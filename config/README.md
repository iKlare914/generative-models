# YAML 任务配置

训练配置放在 `config/train/`，采样配置放在 `config/test/`。每个原有 `.sh`
启动脚本都有一个同名 YAML；脚本只负责选择 Python 入口和配置路径。

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
uv run --no-sync python scripts/diffusion_train.py --config config/train/fm-run.yaml

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
channel_mult: [1, 2, 4]
attention_resolutions: [16, 8]  # [] 关闭各层注意力，bottleneck仍保留
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

## 各方法适用的配置

下表列出算法专用参数；模型结构、设备、种子、路径等公共参数仍正常保留。
YAML 只填写当前方法实际使用的字段，不需要把其他方法的默认值写进去。

| 任务 | 方法 | 算法专用参数 |
| --- | --- | --- |
| 训练（含 CFG） | DDPM | `timesteps` |
| 训练（含 CFG） | FM | `fm_step`、`solver`（用于训练期间记录采样结果） |
| 采样（含 CFG） | DDPM | `timesteps` |
| 采样（含 CFG） | DDIM | `timesteps`、`timestep_spacing`、`randomness` |
| 采样（含 CFG） | FM | `fm_step`、`solver`、`randomness` |

DDIM 使用 DDPM 训练的模型，没有独立的 DDIM 训练配置。DDPM 的随机噪声由
其反向过程决定，不读取 `randomness`；该字段只控制 DDIM 的 eta 或 FM 的随机强度。

CFG 配置额外包含 `feature_channels`、`text_model_name`、`eval_max_length`、
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

例如 `timesteps: 1000`、`timestep_spacing: 20` 对应 51 个 DDIM 步骤，
`skip_steps: 5` 后实际执行 46 次更新。现有 CFG DDPM/DDIM 配置保留的
`skip_steps: 5` 现在也会生效；如需完整时间表，改为 `0`。
