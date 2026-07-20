import os

# 必须放在导入 transformers 之前
# 禁止 Transformers 加载 TensorFlow 和 Flax，只使用 PyTorch
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"

import time
import traceback

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# 路径设置
# ============================================================

# 当前脚本目录：
# /root/private_data/romanian-bert-weather-horoscope/scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型项目根目录：
# /root/private_data/romanian-bert-weather-horoscope
ROOT_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..")
)

# 配置和Tokenizer文件所在目录
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

# 模型权重所在目录
WEIGHT_DIR = os.path.join(ROOT_DIR, "weight")


# 标签含义
LABEL_MEANING = {
    0: "正面",
    1: "负面",
}


# 四条罗马尼亚语测试文本
TEST_SAMPLES = [
    {
        "type": "正面天气文本",
        "expected": "正面",
        "text": (
            "Mâine vremea va fi frumoasă și însorită, "
            "cu temperaturi plăcute și fără precipitații."
        ),
        "translation": "明天天气晴朗，气温宜人，没有降水。",
    },
    {
        "type": "负面天气文本",
        "expected": "负面",
        "text": (
            "Mâine vor fi ploi torențiale, furtuni puternice "
            "și rafale periculoase de vânt."
        ),
        "translation": "明天将有暴雨、强烈风暴和危险阵风。",
    },
    {
        "type": "正面星座文本",
        "expected": "正面",
        "text": (
            "Astăzi vei avea succes, energie pozitivă "
            "și vei primi vești bune."
        ),
        "translation": "今天你会取得成功，充满积极能量，并收到好消息。",
    },
    {
        "type": "负面星座文本",
        "expected": "负面",
        "text": (
            "Astăzi pot apărea conflicte, pierderi "
            "și multe dezamăgiri."
        ),
        "translation": "今天可能出现冲突、损失和许多失望。",
    },
]


def synchronize(device):
    """等待DCU/GPU计算完成，使推理计时更准确。"""
    if device.type == "cuda":
        torch.cuda.synchronize()


def check_model_files():
    """检查模型推理需要的文件。"""

    required_files = [
        os.path.join(CONFIG_DIR, "config.json"),
        os.path.join(CONFIG_DIR, "special_tokens_map.json"),
        os.path.join(CONFIG_DIR, "tokenizer_config.json"),
        os.path.join(CONFIG_DIR, "vocab.txt"),
        os.path.join(WEIGHT_DIR, "model.safetensors"),
    ]

    print("\n" + "=" * 70)
    print("1. 检查模型文件")
    print("=" * 70)

    for file_path in required_files:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"缺少模型文件：{file_path}"
            )

        size_mb = os.path.getsize(file_path) / 1024**2
        relative_path = os.path.relpath(
            file_path,
            ROOT_DIR,
        )

        print(
            f"{relative_path}：存在，"
            f"大小 {size_mb:.2f} MB"
        )


def main():
    print("=" * 70)
    print("Romanian BERT 模型测试")
    print("=" * 70)

    print("项目根目录：", ROOT_DIR)
    print("配置目录：", CONFIG_DIR)
    print("权重目录：", WEIGHT_DIR)
    print("PyTorch版本：", torch.__version__)
    print("CUDA/DCU是否可用：", torch.cuda.is_available())

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("运行设备：", device)

    if device.type == "cuda":
        print("设备数量：", torch.cuda.device_count())
        print("设备名称：", torch.cuda.get_device_name(0))

        total_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )

        print(f"设备总显存：{total_memory:.2f} GB")

    else:
        print("警告：当前使用CPU进行推理")

    check_model_files()

    # ========================================================
    # 加载Tokenizer
    # ========================================================
    print("\n" + "=" * 70)
    print("2. 加载Tokenizer")
    print("=" * 70)

    tokenizer_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG_DIR,
        local_files_only=True,
    )

    tokenizer_time = (
        time.perf_counter() - tokenizer_start
    )

    print("Tokenizer加载成功")
    print("Tokenizer类型：", tokenizer.__class__.__name__)
    print(f"Tokenizer加载时间：{tokenizer_time:.4f} 秒")

    # ========================================================
    # 加载模型配置
    # ========================================================
    print("\n" + "=" * 70)
    print("3. 加载模型配置")
    print("=" * 70)

    config = AutoConfig.from_pretrained(
        CONFIG_DIR,
        local_files_only=True,
    )

    print("模型配置加载成功")
    print("模型类型：", config.model_type)
    print("分类数量：", config.num_labels)
    print("标签配置：", config.id2label)

    # ========================================================
    # 加载模型权重
    # ========================================================
    print("\n" + "=" * 70)
    print("4. 加载模型权重")
    print("=" * 70)

    model_start = time.perf_counter()

    model = (
        AutoModelForSequenceClassification.from_pretrained(
            WEIGHT_DIR,
            config=config,
            local_files_only=True,
        )
    )

    model = model.to(device)
    model.eval()

    synchronize(device)

    model_load_time = (
        time.perf_counter() - model_start
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("模型加载成功")
    print("模型类型：", model.__class__.__name__)
    print("模型所在设备：", next(model.parameters()).device)
    print("模型参数量：", f"{parameter_count:,}")
    print("分类数量：", model.config.num_labels)
    print("标签配置：", model.config.id2label)
    print(f"模型加载时间：{model_load_time:.4f} 秒")

    # ========================================================
    # 构造模型输入
    # ========================================================
    print("\n" + "=" * 70)
    print("5. 构造模型输入")
    print("=" * 70)

    texts = [
        sample["text"]
        for sample in TEST_SAMPLES
    ]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    print("输入文本数量：", len(texts))
    print(
        "input_ids形状：",
        tuple(inputs["input_ids"].shape),
    )
    print(
        "attention_mask形状：",
        tuple(inputs["attention_mask"].shape),
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # ========================================================
    # 模型推理
    # ========================================================
    print("\n" + "=" * 70)
    print("6. 开始模型推理")
    print("=" * 70)

    # 预热一次
    with torch.inference_mode():
        _ = model(**inputs)

    synchronize(device)

    inference_start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(**inputs)

    synchronize(device)

    inference_time = (
        time.perf_counter() - inference_start
    )

    logits = outputs.logits

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    predicted_ids = torch.argmax(
        probabilities,
        dim=-1,
    )

    print("logits形状：", tuple(logits.shape))
    print(f"批量推理时间：{inference_time:.6f} 秒")
    print(
        "平均每条文本推理时间："
        f"{inference_time / len(texts):.6f} 秒"
    )

    # ========================================================
    # 输出结果
    # ========================================================
    print("\n" + "=" * 70)
    print("7. 推理结果")
    print("=" * 70)

    correct_count = 0

    for index, sample in enumerate(TEST_SAMPLES):
        predicted_id = int(
            predicted_ids[index].item()
        )

        predicted_label = model.config.id2label.get(
            predicted_id,
            f"LABEL_{predicted_id}",
        )

        predicted_meaning = LABEL_MEANING.get(
            predicted_id,
            "未知类别",
        )

        sample_probabilities = (
            probabilities[index]
            .detach()
            .cpu()
            .tolist()
        )

        is_correct = (
            predicted_meaning == sample["expected"]
        )

        if is_correct:
            correct_count += 1

        print(f"\n测试样本 {index + 1}")
        print("-" * 70)
        print("样本类型：", sample["type"])
        print("人工预期：", sample["expected"])
        print("罗马尼亚语输入：", sample["text"])
        print("中文含义：", sample["translation"])
        print("预测类别编号：", predicted_id)
        print("预测标签：", predicted_label)
        print("预测情感：", predicted_meaning)
        print(
            "是否符合人工预期：",
            "是" if is_correct else "否",
        )
        print("各类别概率：")

        for class_id, probability in enumerate(
            sample_probabilities
        ):
            label = model.config.id2label.get(
                class_id,
                f"LABEL_{class_id}",
            )

            meaning = LABEL_MEANING.get(
                class_id,
                "未知类别",
            )

            print(
                f"  {label}（{meaning}）："
                f"{probability:.6f}"
            )

    print("\n" + "=" * 70)
    print("8. 测试总结")
    print("=" * 70)

    print(
        f"符合人工预期："
        f"{correct_count}/{len(TEST_SAMPLES)}"
    )

    print(
        "说明：这里只是自定义样本功能测试，"
        "不能作为模型正式准确率。"
    )

    print("\n" + "=" * 70)
    print("模型测试完成：模型已成功加载并完成推理")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()

    except Exception:
        print("\n模型测试失败，完整错误如下：")
        traceback.print_exc()
        raise