# predictia/cerra_tas_vqvae 适配测试 — 运行说明

## 这个模型是干什么的
出自 "Europe Reanalysis Super Resolution" 项目(ECMWF Code for Earth 2023 资助),
目标是用深度学习把 ERA5 全球再分析数据降尺度成 CERRA 那样的高分辨率区域再分析数据。
本仓库是其中的一个 VQ-VAE 子模块:对单通道 80×80 的温度场做「编码 → 离散量化 → 解码」重构,
对应 `diffusers` 库的 `VQModel`(不是文本条件的 image-to-image 扩散模型,模型卡自动生成的
`DiffusionPipeline` / `AutoModel` / `pipeline("image-to-image")` 用法对它无效)。

## 目录应有的文件
```
config.json                        模型结构配置(已提供)
diffusion_pytorch_model.bin        权重,约356KB(已随仓库提交)
test.py / run.sh                   测试脚本
vqvae_torch.py                     备用加载后端(diffusers 异常时自动启用)
diagnose.py                        排查脚本(仅出错时使用)
download_weights.py                权重下载脚本(备用)
requirements.txt                   依赖清单
```

## 运行
```bash
cd models/earth/Cerra_tas_vqvae
bash run.sh
```

## 看结果
运行结束后终端会打印一段 `REPORT BLOCK`,同时写入 `result.log` / `result.json`,
把里面的数值写进测试报告即可。

运行还会生成 `reconstruction.png`、`io_arrays.npz`、`flaggems_ops.log` 作为运行证据。

判定标准:模型加载成功、前向推理无算子报错、输出 shape 与输入一致、无 NaN/Inf —— 四项全满足即为 PASS。

## 运行中的告警可以忽略
- `NVMLError_NotSupported` / `skip unavailable vLLM extension`:DCU 没有 NVIDIA NVML,与本模型无关
- `Overriding a previously registered kernel`:FlagGems 接管算子的正常提示
- `memory efficient attention` 警告:只影响速度,不影响正确性
- `Defaulting to unsafe serialization`:权重是 .bin 格式的正常提示

## 跑不通时
```bash
python diagnose.py    # 生成 diagnose.log,发回排查
```
常用开关:`USE_FLAGGEMS=0 bash run.sh`(关闭FlagGems排查算子问题)、`BLOCK_TENSORFLOW=0 bash run.sh`(不屏蔽tensorflow)
