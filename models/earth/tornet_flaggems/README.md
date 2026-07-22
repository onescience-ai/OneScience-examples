# TorNet FlagGems 适配示例

## 模型功能

该模型基于WSR-88D双偏振天气雷达数据，
识别雷达样本中是否存在龙卷特征，
属于气象遥感与极端天气灾害预警领域的二分类模型。

## 模型来源

- 模型名称：TorNet Tornado Detector Baseline
- 模型仓库：tornet-ml/tornado_detector_baseline_v1
- 原始源码：mit-ll/tornet
- 数据来源：TorNet 2013及catalog

## 目录结构

    tornet_flaggems/
    ├── README.md
    ├── RESULTS.md
    ├── download_data.py
    ├── .gitignore
    └── phr-tornet-main/
        ├── phr-TorNet.ipynb
        ├── phr-TorNet.py
        ├── tornet/
        ├── scripts/
        └── ...

## 未提交的运行资源

为避免向代码仓库上传大文件、数据和日志，
本仓库不直接提交：

- catalog.csv
- tornado_detector_baseline.keras
- train/
- test/
- tornet_2013.tar.gz
- gems_debug.log
- 其他运行日志和缓存

这些资源应通过download_data.py下载到本地数据目录。

## 下载模型和catalog

    python download_data.py \
      --output-dir /root/private_data/phr/phr-TorNet-data

## 下载2013年数据

    python download_data.py \
      --output-dir /root/private_data/phr/phr-TorNet-data \
      --with-2013-data \
      --extract \
      --remove-archive

## 运行

准备好数据后执行：

    cd /root/private_data/phr/phr-tornet-main

    export TORNET_ROOT=/root/private_data/phr/phr-TorNet-data

    python -u phr-TorNet.py

## 验证结果

适配和测试结果见RESULTS.md。

本次验证使用2013年测试数据，
不代表2013—2022官方全量基准测试。
