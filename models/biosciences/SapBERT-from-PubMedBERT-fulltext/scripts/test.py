import os

# 只使用 PyTorch，避免 TensorFlow 和 Flax 库冲突
# 必须放在导入 transformers 之前
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"

import time
import traceback

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


# ============================================================
# 路径设置
# ============================================================

# 当前脚本所在的 scripts 目录
SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# SapBERT 项目根目录
ROOT_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..")
)

# 配置和 Tokenizer 文件所在目录
CONFIG_DIR = os.path.join(
    ROOT_DIR,
    "config",
)

# 模型权重所在目录
WEIGHT_DIR = os.path.join(
    ROOT_DIR,
    "weight",
)


# 生物医学实体名称
ALL_NAMES = [
    "myocardial infarction",   # 心肌梗死
    "heart attack",            # 心脏病发作
    "COVID-19",
    "Coronavirus infection",
    "high fever",              # 高烧
    "Hydroxychloroquine",      # 羟氯喹
]


def synchronize(device):
    """等待 GPU/DCU 计算完成，使计时更准确。"""

    if device.type == "cuda":
        torch.cuda.synchronize()


def check_model_files():
    """检查模型推理需要的文件。"""

    required_files = [
        os.path.join(
            CONFIG_DIR,
            "config.json",
        ),
        os.path.join(
            CONFIG_DIR,
            "special_tokens_map.json",
        ),
        os.path.join(
            CONFIG_DIR,
            "tokenizer_config.json",
        ),
        os.path.join(
            CONFIG_DIR,
            "vocab.txt",
        ),
        os.path.join(
            WEIGHT_DIR,
            "model.safetensors",
        ),
    ]

    print("\n" + "=" * 72)
    print("1. 检查模型文件")
    print("=" * 72)

    for file_path in required_files:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"缺少模型文件：{file_path}"
            )

        size_mb = (
            os.path.getsize(file_path)
            / 1024**2
        )

        relative_path = os.path.relpath(
            file_path,
            ROOT_DIR,
        )

        print(
            f"{relative_path}：存在，"
            f"大小 {size_mb:.2f} MB"
        )


def cosine_similarity_matrix(embeddings):
    """计算实体向量之间的余弦相似度。"""

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    # 防止除零
    norms = np.maximum(
        norms,
        1e-12,
    )

    normalized_embeddings = (
        embeddings / norms
    )

    return (
        normalized_embeddings
        @ normalized_embeddings.T
    )


def main():
    print("=" * 72)
    print("SapBERT 生物医学实体向量测试")
    print("=" * 72)

    print("项目根目录：", ROOT_DIR)
    print("配置目录：", CONFIG_DIR)
    print("权重目录：", WEIGHT_DIR)
    print("PyTorch版本：", torch.__version__)
    print(
        "CUDA/DCU是否可用：",
        torch.cuda.is_available(),
    )

    # 选择运行设备
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("运行设备：", device)

    if device.type == "cuda":
        print(
            "设备数量：",
            torch.cuda.device_count(),
        )

        print(
            "设备名称：",
            torch.cuda.get_device_name(0),
        )

        total_memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            / 1024**3
        )

        print(
            f"设备总显存："
            f"{total_memory:.2f} GB"
        )

    else:
        print("警告：当前使用CPU进行推理")

    check_model_files()

    # ========================================================
    # 加载 Tokenizer
    # ========================================================
    print("\n" + "=" * 72)
    print("2. 加载Tokenizer")
    print("=" * 72)

    tokenizer_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG_DIR,
        local_files_only=True,
    )

    tokenizer_time = (
        time.perf_counter()
        - tokenizer_start
    )

    print("Tokenizer加载成功")
    print(
        "Tokenizer类型：",
        tokenizer.__class__.__name__,
    )

    print(
        f"Tokenizer加载时间："
        f"{tokenizer_time:.4f} 秒"
    )

    # ========================================================
    # 加载模型配置
    # ========================================================
    print("\n" + "=" * 72)
    print("3. 加载模型配置")
    print("=" * 72)

    config_start = time.perf_counter()

    config = AutoConfig.from_pretrained(
        CONFIG_DIR,
        local_files_only=True,
    )

    config_time = (
        time.perf_counter()
        - config_start
    )

    print("模型配置加载成功")
    print(
        "配置类型：",
        config.__class__.__name__,
    )
    print(
        "模型类型：",
        config.model_type,
    )
    print(
        "隐藏层维度：",
        config.hidden_size,
    )
    print(
        f"配置加载时间："
        f"{config_time:.4f} 秒"
    )

    # ========================================================
    # 加载模型和权重
    # ========================================================
    print("\n" + "=" * 72)
    print("4. 加载SapBERT模型和权重")
    print("=" * 72)

    model_start = time.perf_counter()

    model = AutoModel.from_pretrained(
        WEIGHT_DIR,
        config=config,
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()

    synchronize(device)

    model_load_time = (
        time.perf_counter()
        - model_start
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("模型加载成功")
    print(
        "模型类型：",
        model.__class__.__name__,
    )
    print(
        "模型所在设备：",
        next(model.parameters()).device,
    )
    print(
        "模型参数量：",
        f"{parameter_count:,}",
    )
    print(
        "隐藏层维度：",
        model.config.hidden_size,
    )
    print(
        f"模型和权重加载时间："
        f"{model_load_time:.4f} 秒"
    )

    # ========================================================
    # 提取实体向量
    # ========================================================
    print("\n" + "=" * 72)
    print("5. 提取实体向量")
    print("=" * 72)

    print(
        "实体数量：",
        len(ALL_NAMES),
    )

    batch_size = 128
    all_embeddings = []

    synchronize(device)

    inference_start = time.perf_counter()

    for start_index in tqdm(
        np.arange(
            0,
            len(ALL_NAMES),
            batch_size,
        ),
        desc="正在提取实体向量",
    ):
        # 对当前批次进行分词
        tokens = tokenizer.batch_encode_plus(
            ALL_NAMES[
                start_index:
                start_index + batch_size
            ],
            padding="max_length",
            max_length=25,
            truncation=True,
            return_tensors="pt",
        )

        # 将输入移动到运行设备
        tokens_device = {
            key: value.to(device)
            for key, value in tokens.items()
        }

        # 模型前向推理
        with torch.inference_mode():
            outputs = model(
                **tokens_device
            )

            # 取最后一层第0个Token的向量
            cls_embeddings = (
                outputs[0][:, 0, :]
            )

        # 转移到CPU并转换为NumPy数组
        all_embeddings.append(
            cls_embeddings
            .cpu()
            .numpy()
        )

    synchronize(device)

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    # 合并所有批次
    all_embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    print(
        "实体向量形状：",
        all_embeddings.shape,
    )

    print(
        f"推理时间："
        f"{inference_time:.6f} 秒"
    )

    print(
        "平均每个实体推理时间："
        f"{inference_time / len(ALL_NAMES):.6f} 秒"
    )

    print(
        "第一个实体向量的前10个值：",
        np.round(
            all_embeddings[0][:10],
            6,
        ),
    )

    # ========================================================
    # 计算余弦相似度
    # ========================================================
    print("\n" + "=" * 72)
    print("6. 计算两两余弦相似度")
    print("=" * 72)

    similarity_matrix = (
        cosine_similarity_matrix(
            all_embeddings
        )
    )

    print(
        "相似度矩阵形状：",
        similarity_matrix.shape,
    )

    print("\n实体编号：")

    for index, name in enumerate(
        ALL_NAMES
    ):
        print(
            f"{index}：{name}"
        )

    print("\n相似度矩阵：")

    print(
        np.round(
            similarity_matrix,
            4,
        )
    )

    # ========================================================
    # 显示重点实体对
    # ========================================================
    print("\n重点对比：")

    comparison_pairs = [
        (0, 1),
        (2, 3),
        (0, 4),
        (1, 5),
    ]

    for left, right in comparison_pairs:
        score = similarity_matrix[
            left,
            right,
        ]

        print(
            f"{ALL_NAMES[left]} "
            f"<-> {ALL_NAMES[right]}："
            f"{score:.6f}"
        )

    print("\n" + "=" * 72)
    print(
        "模型测试完成："
        "成功提取实体向量并计算语义相似度"
    )
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()

    except Exception:
        print(
            "\n模型测试失败，完整错误如下："
        )

        traceback.print_exc()
        raise