# DGMR_SolRad 的复现与适配

太阳辐射临近预报模型 DGMR-SO 在国产 DCU + FlagOS 环境下的复现说明。

- 原模型：<https://huggingface.co/thingnario/DGMR_SolRad>
- 状态：**已复现成功**（FlagGems 启用，输出 `(1,36,512,512)`、无 NaN/Inf、推理耗时 2933.2 ms）

---

## 一、环境

镜像自带 Python 3.10 / torch 2.4.1 / numpy / einops，通常无需安装任何东西。

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.4.1  True
```

---

## 二、本仓库包含什么

本仓库**只含适配代码**，不含模型权重与样本数据（体积超出仓库限制）。

| 文件 | 说明 |
|---|---|
| `README.md` | 本说明 |
| `test.py` | 适配测试脚本（前向与官方 `inference.py` 等价，另加 FlagGems 开关、计时、输出校验） |
| `run.sh` | 一键运行封装 |
| `run_202504131100_DGMR_SO.log` | 实测运行日志（`bash run.sh` 产出），作为复现证据 |
| `flaggems_ops.log` | FlagGems 算子调用日志 |

模型本体（`model_architect/`、`model_weights/`、`sample_data/`，合计约 545MB）
需按下节从 HuggingFace 获取。

---

## 三、准备文件

工作目录 `/root/private_data/DGMR_SolRad`，下载模型仓库（约 545MB）：

```bash
mkdir -p /root/private_data/DGMR_SolRad
cd /root/private_data/DGMR_SolRad
hf download thingnario/DGMR_SolRad --repo-type model --local-dir .
```

再把本包的 `test.py`、`run.sh` 放进同一目录。最终结构：

```
/root/private_data/DGMR_SolRad/
├── inference.py          # 官方推理脚本
├── test.py               # 本包：适配测试脚本
├── run.sh                # 本包：一键运行
├── model_architect/      # 模型结构源码
├── model_weights/        # 权重（DGMR_SO/ft36/weights.ckpt 约 206MB）
└── sample_data/          # 样本输入（3 个 npz，各约 32MB）
```

其中 `inference.py`、`model_architect/`、`model_weights/`、`sample_data/` 来自 `hf download`；
`test.py`、`run.sh` 来自本仓库。

校验：

```bash
du -sh model_weights/DGMR_SO/ft36/weights.ckpt   # 应约 206M
du -sh sample_data/*.npz                          # 应各约 32M
```

---

## 四、运行

```bash
cd /root/private_data/DGMR_SolRad
bash run.sh DGMR_SO 202504131100
```

会自动检查文件、依赖，运行后落日志 `run_202504131100_DGMR_SO.log`。

或直接调用脚本（参数可省略，默认即下列值）：

```bash
python test.py --model-type DGMR_SO --basetime 202504131100
```

**可选参数**

| 写法 | 作用 |
|---|---|
| `--model-type` | `DGMR_SO`（默认）或 `Generator_only` |
| `--basetime` | `202504131100`（默认）/ `202504161200` / `202507151200` |
| `USE_FLAGGEMS=0` | 关闭 FlagGems，用 DCU 原生 torch |
| `SEED=0` | 固定随机种子，使 DGMR_SO 输出可复现 |

**输出产物**

- `pred_{basetime}_{model_type}.npy` — 预测结果，形状 `(1,36,512,512)`，单位 W/m²
- `run_*.log` / `flaggems_ops.log` — 运行日志、FlagGems 算子调用记录

---

## 五、FlagGems 算子回退

`test.py` 中的 `FLAGGEMS_UNUSED` 已填好实测确认需回退的算子，**保持原样即可**：

| 算子 | 回退原因 |
|---|---|
| `pixel_unshuffle` / `pixel_shuffle` | FlagGems 版仅支持 4D，本模型输入为 5D `(N,D,C,H,W)` |
| `reflection_pad2d` | ConvGRU 的 reflect padding |
| `mv` / `dot` | spectral_norm 功率迭代；Triton autotune 开销极大且此处无收益 |

回退不影响计算正确性，其余算子仍由 FlagGems 接管。若换平台后报 `xxx is not supported in flag_gems`，把该算子名加进列表重跑即可。

---

## 六、说明

**输出是随机的。** DGMR_SO 内部对隐变量做高斯采样，同一输入多次运行结果不同，属正常现象（Generator_only 无采样，输出确定）。因此测试以「跑通 + 输出校验（形状正确、无 NaN/Inf、数值合理）」为准，而非数值一致。需要复现时加 `SEED=0`。

**无误差指标。** 样本不含未来真值，故不计算 MAE/RMSE。若样本含真值键，`test.py` 会自动计算。

---

## 七、常见问题

| 现象 | 处理 |
|---|---|
| `FileNotFoundError: *.npz / *.ckpt` | 文件没下全，重跑 `hf download` 并校验文件大小 |
| `ModuleNotFoundError: model_architect` | 没在仓库根目录运行，或目录层级被打乱 |
| 长时间无输出、无报错 | FlagGems 首次遇到新算子形状时会做 Triton autotune，属正常现象，耐心等待 |
| 两次输出数值不同 | 正常（随机隐变量），加 `SEED=0` 固定 |
