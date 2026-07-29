#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""dazzle-nu/autotrain-weather-classification-3739699407 平台复现脚本。"""
import sys
import subprocess

# 不安装 torch，避免覆盖平台自带的 GPU/ROCm 适配版本。
packages = [
    "numpy==1.26.3",
    "transformers==4.46.3",
    "huggingface_hub==0.36.0",
    "datasets==2.21.0",
    "pillow==10.4.0",
    "matplotlib==3.9.2",
    "scikit-learn==1.5.2",
    "pandas==2.2.2",
    "tqdm==4.66.6",
]

cmd = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-cache-dir",
    "--upgrade",
    *packages,
]

print("执行：")
print(" ".join(cmd))
subprocess.check_call(cmd)

print("\n依赖安装完成。")
print("继续执行模型复现流程。")

# %% 代码单元 2
import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MPLBACKEND"] = "Agg"

import sys
import json
import io
import gc
import shutil
import random
import platform
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import transformers
import datasets
import huggingface_hub
import matplotlib
import matplotlib.pyplot as plt
import sklearn

from PIL import Image
from tqdm.auto import tqdm

warnings.filterwarnings("default")

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("Hugging Face Hub:", huggingface_hub.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Scikit-learn:", sklearn.__version__)
print("USE_TF:", os.environ["USE_TF"])
print("USE_FLAX:", os.environ["USE_FLAX"])
print("USE_TORCH:", os.environ["USE_TORCH"])

assert np.__version__ == "1.26.3", (
    f"当前 NumPy 为 {np.__version__}，应为 1.26.3。"
    "请重新运行第 1 步并重启 Kernel。"
)

# 验证 NumPy 与 PyTorch 能够双向转换。
test_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
test_tensor = torch.from_numpy(test_array)
test_array_back = test_tensor.numpy()

print("\nNumPy → PyTorch:", test_tensor)
print("PyTorch → NumPy:", test_array_back)
print("环境导入和 NumPy/PyTorch 兼容性检查通过。")


# ==============================================================================
# 第2步：环境变量、依赖导入与兼容性检查
# ==============================================================================

# %% 代码单元 3
# 个人目录设置
PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR.parent

MODEL_DIR = PROJECT_DIR / "model"
DATASET_DIR = PROJECT_DIR / "weather-classification-data"
RESULT_DIR = PROJECT_DIR / "results"
HF_CACHE_DIR = PROJECT_DIR / ".hf_cache"

for path in [
    BASE_DIR,
    PROJECT_DIR,
    MODEL_DIR,
    DATASET_DIR,
    RESULT_DIR,
    HF_CACHE_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Hugging Face 缓存设置
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")
os.environ["HF_HUB_DISABLE_XET"] = "1"

MODEL_ID = "dazzle-nu/autotrain-weather-classification-3739699407"
DATASET_ID = "dazzle-nu/autotrain-data-weather-classification"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("MODEL_ID    :", MODEL_ID)
print("DATASET_ID  :", DATASET_ID)
print("BASE_DIR    :", BASE_DIR)
print("PROJECT_DIR :", PROJECT_DIR)
print("MODEL_DIR   :", MODEL_DIR)
print("DATASET_DIR :", DATASET_DIR)
print("RESULT_DIR  :", RESULT_DIR)
print("HF_CACHE_DIR:", HF_CACHE_DIR)
print("SEED        :", SEED)


# ==============================================================================
# 第3步：设置目录、缓存和随机种子
# ==============================================================================

# %% 代码单元 4
def bytes_to_gib(value):
    return value / (1024 ** 3)

total, used, free = shutil.disk_usage(BASE_DIR)

print(f"磁盘总空间：{bytes_to_gib(total):.2f} GiB")
print(f"磁盘已使用：{bytes_to_gib(used):.2f} GiB")
print(f"磁盘可用空间：{bytes_to_gib(free):.2f} GiB")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("\n加速设备可用：", torch.cuda.get_device_name(0))
    print("设备数量：", torch.cuda.device_count())
    try:
        properties = torch.cuda.get_device_properties(0)
        print(f"设备显存：{bytes_to_gib(properties.total_memory):.2f} GiB")
    except Exception as exc:
        print("未能读取显存信息：", repr(exc))
else:
    DEVICE = torch.device("cpu")
    print("\n未检测到 torch.cuda 加速设备，将使用 CPU。")

print("最终使用设备：", DEVICE)

assert free > 1.2 * 1024 ** 3, "可用空间不足，建议至少保留 1.2 GiB。"


# ==============================================================================
# 第4步：检查设备和磁盘空间
# ==============================================================================

# %% 代码单元 5
from huggingface_hub import snapshot_download

model_snapshot = snapshot_download(
    repo_id=MODEL_ID,
    revision="main",
    local_dir=str(MODEL_DIR),
    cache_dir=str(HF_CACHE_DIR / "hub"),
)

print("模型下载目录：", model_snapshot)
print("\n模型目录文件：")

for path in sorted(MODEL_DIR.iterdir()):
    if path.is_file():
        print(f"- {path.name}: {path.stat().st_size / (1024 ** 2):.2f} MiB")

required_model_files = [
    MODEL_DIR / "config.json",
    MODEL_DIR / "preprocessor_config.json",
    MODEL_DIR / "pytorch_model.bin",
]

missing_model_files = [
    str(path)
    for path in required_model_files
    if not path.exists() or path.stat().st_size == 0
]

assert not missing_model_files, (
    f"模型文件缺失或为空：{missing_model_files}"
)

print("\n模型文件完整性检查通过。")


# ==============================================================================
# 第5步：下载并校验模型
# ==============================================================================

# %% 代码单元 6
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

processor = AutoImageProcessor.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)

# 强制使用 eager 注意力。
# 已验证 eager 与默认实现的评估结果一致。
model = AutoModelForImageClassification.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
    attn_implementation="eager",
)

model.to(DEVICE)
model.eval()

num_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("模型类型：", model.__class__.__name__)
print("模型参数量：", f"{num_parameters:,}")
print("可训练参数量：", f"{trainable_parameters:,}")
print("模型所在设备：", next(model.parameters()).device)
print(
    "注意力实现：",
    getattr(model.config, "_attn_implementation", "未显示"),
)
print("输入尺寸：", processor.size)
print("类别数量：", model.config.num_labels)
print("类别映射：", model.config.id2label)

assert model.__class__.__name__ == "ViTForImageClassification"
assert model.config.num_labels == 11
assert int(processor.size["height"]) == 224
assert int(processor.size["width"]) == 224

print("\n模型加载和结构检查通过。")


# ==============================================================================
# 第6步：加载模型
# ==============================================================================

# %% 代码单元 7
from huggingface_hub import snapshot_download

dataset_snapshot = snapshot_download(
    repo_id=DATASET_ID,
    repo_type="dataset",
    revision="main",
    local_dir=str(DATASET_DIR),
    cache_dir=str(HF_CACHE_DIR / "hub"),
    allow_patterns=[
        "processed/valid/dataset.arrow",
        "processed/valid/dataset_info.json",
        "processed/valid/state.json",
        "processed/dataset_dict.json",
        "README.md",
    ],
)

VALID_DIR = DATASET_DIR / "processed" / "valid"

print("数据集下载目录：", dataset_snapshot)
print("验证集目录：", VALID_DIR)

for path in sorted(VALID_DIR.iterdir()):
    if path.is_file():
        print(f"- {path.name}: {path.stat().st_size / (1024 ** 2):.2f} MiB")

required_dataset_files = [
    VALID_DIR / "dataset.arrow",
    VALID_DIR / "dataset_info.json",
    VALID_DIR / "state.json",
]

missing_dataset_files = [
    str(path)
    for path in required_dataset_files
    if not path.exists() or path.stat().st_size == 0
]

assert not missing_dataset_files, (
    f"验证集文件缺失或为空：{missing_dataset_files}"
)

print("\n验证集文件完整性检查通过。")


# ==============================================================================
# 第7步：下载并校验验证集
# ==============================================================================

# %% 代码单元 8
from datasets import load_from_disk

valid_ds = load_from_disk(str(VALID_DIR))

print(valid_ds)
print("验证集样本数量：", len(valid_ds))
print("字段：", valid_ds.column_names)
print("特征定义：", valid_ds.features)

assert len(valid_ds) == 1378, (
    f"验证集数量异常：{len(valid_ds)}，预期为 1378。"
)

assert "image" in valid_ds.column_names
assert "target" in valid_ds.column_names

dataset_labels = list(
    valid_ds.features["target"].names
)

model_labels = [
    model.config.id2label[class_id]
    for class_id in range(model.config.num_labels)
]

print("\n数据集标签：", dataset_labels)
print("模型标签：", model_labels)
print("是否完全一致：", dataset_labels == model_labels)

assert dataset_labels == model_labels, (
    "数据集标签顺序与模型标签顺序不一致。"
)

print("\n验证集数量和标签顺序检查通过。")


# ==============================================================================
# 第8步：加载验证集并核对标签
# ==============================================================================

# %% 代码单元 9
def ensure_pil_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(
                io.BytesIO(value["bytes"])
            ).convert("RGB")

        if value.get("path"):
            return Image.open(
                value["path"]
            ).convert("RGB")

    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")

    raise TypeError(
        f"无法转换为 PIL 图片，实际类型为：{type(value)}"
    )


sample_index = 0
sample = valid_ds[sample_index]
sample_image = ensure_pil_image(sample["image"])
true_id = int(sample["target"])
true_label = model.config.id2label[true_id]

plt.figure(figsize=(7, 5))
plt.imshow(sample_image)
plt.axis("off")
plt.title(
    f"Validation sample {sample_index} | true label: {true_label}"
)
sample_preview_path = RESULT_DIR / "sample_preview.png"
plt.savefig(sample_preview_path, dpi=160, bbox_inches="tight")
plt.close()
print("样例图片：", sample_preview_path)

print("图片尺寸：", sample_image.size)
print("真实类别 ID：", true_id)
print("真实类别：", true_label)


# ==============================================================================
# 第9步：显示验证图片
# ==============================================================================

# %% 代码单元 10
@torch.inference_mode()
def predict_image(image, top_k=5):
    image = ensure_pil_image(image)

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    outputs = model(**inputs)
    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    top_k = min(
        int(top_k),
        probabilities.numel(),
    )

    top_probabilities, top_ids = torch.topk(
        probabilities,
        k=top_k,
    )

    results = []

    for probability, class_id in zip(
        top_probabilities.cpu().tolist(),
        top_ids.cpu().tolist(),
    ):
        results.append(
            {
                "class_id": int(class_id),
                "label": model.config.id2label[int(class_id)],
                "score": float(probability),
            }
        )

    return results


single_results = predict_image(
    sample_image,
    top_k=5,
)

print("真实类别：", true_label)
print("\nTop-5 预测：")

for rank, item in enumerate(
    single_results,
    start=1,
):
    print(
        f"{rank}. {item['label']:<12s} "
        f"ID={item['class_id']:<2d} "
        f"概率={item['score']:.4%}"
    )

predicted_label = single_results[0]["label"]

print("\nTop-1 预测：", predicted_label)
print(
    "该样本预测是否正确：",
    predicted_label == true_label,
)

assert len(single_results) == 5

print("\n单图推理流程检查通过。")


# ==============================================================================
# 第10步：单图推理
# ==============================================================================

# %% 代码单元 11
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

def collate_weather_batch(examples):
    images = [
        ensure_pil_image(example["image"])
        for example in examples
    ]

    labels = torch.tensor(
        [
            int(example["target"])
            for example in examples
        ],
        dtype=torch.long,
    )

    pixel_values = processor(
        images=images,
        return_tensors="pt",
    )["pixel_values"]

    return pixel_values, labels


@torch.inference_mode()
def evaluate_dataset(
    dataset,
    batch_size=32,
    limit=None,
):
    if limit is not None:
        limit = min(
            int(limit),
            len(dataset),
        )
        dataset = dataset.select(
            range(limit)
        )

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collate_weather_batch,
    )

    y_true = []
    y_pred = []
    y_prob = []

    for pixel_values, labels in tqdm(
        loader,
        desc="Evaluating",
    ):
        pixel_values = pixel_values.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(
            pixel_values=pixel_values
        ).logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        predictions = probabilities.argmax(
            dim=-1
        )

        y_true.extend(labels.tolist())
        y_pred.extend(
            predictions.cpu().tolist()
        )
        y_prob.extend(
            probabilities.cpu().tolist()
        )

    return {
        "num_samples": len(y_true),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
            )
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


print("批量评估函数创建完成。")


# ==============================================================================
# 第11步：定义批量评估函数
# ==============================================================================

# %% 代码单元 12
quick_metrics = evaluate_dataset(
    valid_ds,
    batch_size=32 if DEVICE.type == "cuda" else 8,
    limit=200,
)

quick_summary = {
    key: value
    for key, value in quick_metrics.items()
    if not key.startswith("y_")
}

print(
    json.dumps(
        quick_summary,
        indent=2,
        ensure_ascii=False,
    )
)

assert quick_metrics["num_samples"] == 200

print("\n200 张快速评估完成。")


# ==============================================================================
# 第12步：200张快速评估
# ==============================================================================

# %% 代码单元 13
full_metrics = evaluate_dataset(
    valid_ds,
    batch_size=32 if DEVICE.type == "cuda" else 8,
    limit=None,
)

PUBLISHED_METRICS = {
    "accuracy": 0.952,
    "macro_f1": 0.957,
    "weighted_f1": 0.952,
}

summary = {
    "model_id": MODEL_ID,
    "dataset_id": DATASET_ID,
    "device": str(DEVICE),
    "attention_implementation": getattr(
        model.config,
        "_attn_implementation",
        "unknown",
    ),
    "num_samples": full_metrics["num_samples"],
    "accuracy": full_metrics["accuracy"],
    "macro_f1": full_metrics["macro_f1"],
    "weighted_f1": full_metrics["weighted_f1"],
    "published_accuracy": PUBLISHED_METRICS["accuracy"],
    "published_macro_f1": PUBLISHED_METRICS["macro_f1"],
    "published_weighted_f1": PUBLISHED_METRICS["weighted_f1"],
}

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)

assert full_metrics["num_samples"] == 1378

print("\n1378 张验证图片已全部完成推理。")


# ==============================================================================
# 第13步：1378张完整评估
# ==============================================================================

# %% 代码单元 14
METRIC_TOLERANCE = 0.01
metric_comparison = {}

for metric_name, published_value in PUBLISHED_METRICS.items():
    local_value = full_metrics[metric_name]
    difference = abs(
        local_value - published_value
    )
    passed = difference <= METRIC_TOLERANCE

    metric_comparison[metric_name] = {
        "local": float(local_value),
        "published": float(published_value),
        "absolute_difference": float(difference),
        "tolerance": METRIC_TOLERANCE,
        "passed": bool(passed),
    }

    print(
        f"{metric_name}: "
        f"本地={local_value:.6f}, "
        f"公开={published_value:.6f}, "
        f"差值={difference:.6f}, "
        f"{'通过' if passed else '未通过'}"
    )

metric_reproduction_passed = all(
    item["passed"]
    for item in metric_comparison.values()
)

print(
    "\n模型卡公开指标严格复现：",
    "通过"
    if metric_reproduction_passed
    else "未通过",
)


# ==============================================================================
# 第14步：对比公开指标
# ==============================================================================

# %% 代码单元 15
labels = list(
    range(model.config.num_labels)
)

target_names = [
    model.config.id2label[class_id]
    for class_id in labels
]

report_dict = classification_report(
    full_metrics["y_true"],
    full_metrics["y_pred"],
    labels=labels,
    target_names=target_names,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(
    report_dict
).transpose()

print(report_df.to_string())

cm = confusion_matrix(
    full_metrics["y_true"],
    full_metrics["y_pred"],
    labels=labels,
)

fig, ax = plt.subplots(
    figsize=(11, 10)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names,
)

disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d",
    colorbar=False,
)

ax.set_title(
    "Weather Classification Validation Confusion Matrix"
)

fig.tight_layout()
plt.close(fig)


# ==============================================================================
# 第15步：分类报告和混淆矩阵
# ==============================================================================

# %% 代码单元 16
cm_without_diagonal = cm.copy()
np.fill_diagonal(
    cm_without_diagonal,
    0,
)

error_pairs = []

for true_id in range(
    model.config.num_labels
):
    for pred_id in range(
        model.config.num_labels
    ):
        count = int(
            cm_without_diagonal[
                true_id,
                pred_id,
            ]
        )

        if count > 0:
            error_pairs.append(
                (
                    count,
                    true_id,
                    pred_id,
                )
            )

error_pairs.sort(reverse=True)

print("错分数量最多的前 15 组：")

for count, true_id, pred_id in error_pairs[:15]:
    print(
        f"{model.config.id2label[true_id]:<12s} → "
        f"{model.config.id2label[pred_id]:<12s}: "
        f"{count} 张"
    )


# ==============================================================================
# 第16步：主要错分类别
# ==============================================================================

# %% 代码单元 17
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

summary_path = (
    RESULT_DIR / "evaluation_summary.json"
)
comparison_path = (
    RESULT_DIR / "metric_comparison.json"
)
report_path = (
    RESULT_DIR / "classification_report.csv"
)
predictions_path = (
    RESULT_DIR / "validation_predictions.csv"
)
confusion_matrix_path = (
    RESULT_DIR / "confusion_matrix.png"
)
environment_path = (
    RESULT_DIR / "environment.json"
)
status_path = (
    RESULT_DIR / "reproduction_status.json"
)
report_markdown_path = (
    RESULT_DIR / "reproduction_report.md"
)

# 1. 指标汇总
summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

# 2. 指标比较
comparison_path.write_text(
    json.dumps(
        metric_comparison,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

# 3. 分类报告
report_df.to_csv(
    report_path,
    encoding="utf-8-sig",
)

# 4. 逐样本预测
prediction_rows = []

for index, (
    true_id,
    pred_id,
    probabilities,
) in enumerate(
    zip(
        full_metrics["y_true"],
        full_metrics["y_pred"],
        full_metrics["y_prob"],
    )
):
    prediction_rows.append(
        {
            "index": index,
            "true_id": int(true_id),
            "true_label": model.config.id2label[
                int(true_id)
            ],
            "pred_id": int(pred_id),
            "pred_label": model.config.id2label[
                int(pred_id)
            ],
            "correct": bool(
                true_id == pred_id
            ),
            "confidence": float(
                max(probabilities)
            ),
        }
    )

predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_df.to_csv(
    predictions_path,
    index=False,
    encoding="utf-8-sig",
)

# 5. 保存混淆矩阵
fig, ax = plt.subplots(
    figsize=(11, 10)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names,
)

disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d",
    colorbar=False,
)

ax.set_title(
    "Weather Classification Validation Confusion Matrix"
)

fig.tight_layout()

fig.savefig(
    confusion_matrix_path,
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)

# 6. 保存环境信息
environment = {
    "created_at": datetime.now().isoformat(
        timespec="seconds"
    ),
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "pandas": pd.__version__,
    "matplotlib": matplotlib.__version__,
    "scikit_learn": sklearn.__version__,
    "device": str(DEVICE),
    "device_name": (
        torch.cuda.get_device_name(0)
        if DEVICE.type == "cuda"
        else "CPU"
    ),
    "attention_implementation": getattr(
        model.config,
        "_attn_implementation",
        "unknown",
    ),
    "seed": SEED,
}

environment_path.write_text(
    json.dumps(
        environment,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

# 7. 分开判断部署复现和指标复现
deployment_reproduction_passed = (
    all(
        path.exists()
        and path.stat().st_size > 0
        for path in required_model_files
    )
    and model.__class__.__name__
    == "ViTForImageClassification"
    and model.config.num_labels == 11
    and dataset_labels == model_labels
    and len(valid_ds) == 1378
    and full_metrics["num_samples"] == 1378
    and len(predictions_df) == 1378
)

if (
    deployment_reproduction_passed
    and metric_reproduction_passed
):
    conclusion = (
        "模型部署、推理和模型卡公开指标均复现成功。"
    )
elif deployment_reproduction_passed:
    conclusion = (
        "模型部署与推理复现成功；"
        "模型卡公开指标严格复现未通过。"
    )
else:
    conclusion = (
        "模型部署或推理复现尚未完成。"
    )

reproduction_status = {
    "deployment_and_inference_reproduction_passed": bool(
        deployment_reproduction_passed
    ),
    "published_metric_reproduction_passed": bool(
        metric_reproduction_passed
    ),
    "metric_tolerance": METRIC_TOLERANCE,
    "conclusion": conclusion,
}

status_path.write_text(
    json.dumps(
        reproduction_status,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

# 8. 生成 Markdown 报告
report_markdown = f"""# 天气分类模型复现报告

## 模型

`{MODEL_ID}`

## 数据集

`{DATASET_ID}`

## 运行环境

- Python：{sys.version.split()[0]}
- NumPy：{np.__version__}
- PyTorch：{torch.__version__}
- Transformers：{transformers.__version__}
- 设备：{DEVICE}
- 设备名称：{environment['device_name']}
- 注意力实现：{environment['attention_implementation']}

## 本地评估结果

- 样本数量：{full_metrics['num_samples']}
- Accuracy：{full_metrics['accuracy']:.6f}
- Macro F1：{full_metrics['macro_f1']:.6f}
- Weighted F1：{full_metrics['weighted_f1']:.6f}

## 模型卡公开指标

- Accuracy：{PUBLISHED_METRICS['accuracy']:.6f}
- Macro F1：{PUBLISHED_METRICS['macro_f1']:.6f}
- Weighted F1：{PUBLISHED_METRICS['weighted_f1']:.6f}

## 复现判断

- 模型部署与推理复现：{'通过' if deployment_reproduction_passed else '未通过'}
- 模型卡公开指标严格复现：{'通过' if metric_reproduction_passed else '未通过'}

## 最终结论

{conclusion}

本次实验复现的是公开权重加载、公开验证集推理和本地指标计算。原仓库没有公开完整 AutoTrain 训练与评估代码，因此如果本地指标与模型卡指标不一致，应如实记录差异，不能把模型部署成功等同于模型卡指标严格复现成功。
"""

report_markdown_path.write_text(
    report_markdown,
    encoding="utf-8",
)

print("结果已保存：")

for path in [
    summary_path,
    comparison_path,
    report_path,
    predictions_path,
    confusion_matrix_path,
    environment_path,
    status_path,
    report_markdown_path,
]:
    print(
        f"- {path.name}: "
        f"{path.stat().st_size} bytes"
    )


# ==============================================================================
# 第17步：保存结果
# ==============================================================================

# %% 代码单元 18
expected_result_files = [
    RESULT_DIR / "evaluation_summary.json",
    RESULT_DIR / "metric_comparison.json",
    RESULT_DIR / "classification_report.csv",
    RESULT_DIR / "validation_predictions.csv",
    RESULT_DIR / "confusion_matrix.png",
    RESULT_DIR / "environment.json",
    RESULT_DIR / "reproduction_status.json",
    RESULT_DIR / "reproduction_report.md",
]

check_results = {
    "模型文件完整": all(
        path.exists()
        and path.stat().st_size > 0
        for path in required_model_files
    ),
    "模型类型正确": (
        model.__class__.__name__
        == "ViTForImageClassification"
    ),
    "类别数量为 11": (
        model.config.num_labels == 11
    ),
    "模型与数据集标签一致": (
        dataset_labels == model_labels
    ),
    "验证集数量为 1378": (
        len(valid_ds) == 1378
    ),
    "完成 1378 张推理": (
        full_metrics["num_samples"] == 1378
    ),
    "预测表包含 1378 条记录": (
        len(predictions_df) == 1378
    ),
    "结果文件全部生成": all(
        path.exists()
        and path.stat().st_size > 0
        for path in expected_result_files
    ),
    "Accuracy 接近公开值": (
        metric_comparison["accuracy"]["passed"]
    ),
    "Macro F1 接近公开值": (
        metric_comparison["macro_f1"]["passed"]
    ),
    "Weighted F1 接近公开值": (
        metric_comparison[
            "weighted_f1"
        ]["passed"]
    ),
}

print("=" * 72)
print("天气分类模型复现最终验收")
print("=" * 72)

for item, passed in check_results.items():
    print(
        f"{'✅' if passed else '❌'} "
        f"{item}"
    )

deployment_checks = [
    "模型文件完整",
    "模型类型正确",
    "类别数量为 11",
    "模型与数据集标签一致",
    "验证集数量为 1378",
    "完成 1378 张推理",
    "预测表包含 1378 条记录",
    "结果文件全部生成",
]

metric_checks = [
    "Accuracy 接近公开值",
    "Macro F1 接近公开值",
    "Weighted F1 接近公开值",
]

deployment_final = all(
    check_results[item]
    for item in deployment_checks
)

metrics_final = all(
    check_results[item]
    for item in metric_checks
)

print("\n" + "=" * 72)
print(
    "模型部署与推理复现：",
    "通过"
    if deployment_final
    else "未通过",
)

print(
    "模型卡公开指标严格复现：",
    "通过"
    if metrics_final
    else "未通过",
)

if deployment_final and metrics_final:
    final_conclusion = (
        "模型部署、推理和模型卡公开指标均复现成功。"
    )
elif deployment_final:
    final_conclusion = (
        "模型部署与推理复现成功；"
        "模型卡公开指标严格复现未通过。"
    )
else:
    final_conclusion = (
        "模型部署或推理复现尚未完成。"
    )

print("最终结论：", final_conclusion)
print("=" * 72)
