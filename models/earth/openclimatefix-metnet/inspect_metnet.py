import json
import traceback
from pathlib import Path

import torch


MODEL_DIR = Path("hf_snapshot")
CONFIG_PATH = MODEL_DIR / "config.json"
WEIGHT_PATH = MODEL_DIR / "pytorch_model.bin"


def print_title(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


print_title("1. 仓库文件检查")

for path in sorted(MODEL_DIR.rglob("*")):
    if path.is_file():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"{path.relative_to(MODEL_DIR)}: {size_mb:.6f} MB")


print_title("2. config.json 内容")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_dict = json.load(f)

print(json.dumps(config_dict, indent=2, ensure_ascii=False))


print_title("3. Transformers AutoConfig / AutoModel 测试")

try:
    import transformers
    from transformers import AutoConfig, AutoModel

    print("transformers version:", transformers.__version__)

    for trust_remote_code in [False, True]:
        print("\n" + "-" * 80)
        print("trust_remote_code =", trust_remote_code)

        try:
            auto_config = AutoConfig.from_pretrained(
                str(MODEL_DIR),
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
            print("AutoConfig 加载成功")
            print(auto_config)

            try:
                model = AutoModel.from_pretrained(
                    str(MODEL_DIR),
                    trust_remote_code=trust_remote_code,
                    local_files_only=True,
                )
                print("AutoModel 加载成功")
                print("model class:", model.__class__.__name__)
                print(
                    "parameter count:",
                    sum(p.numel() for p in model.parameters()),
                )
            except Exception as e:
                print("AutoModel 加载失败")
                print("error type:", type(e).__name__)
                print("error message:", str(e))
                traceback.print_exc()

        except Exception as e:
            print("AutoConfig 加载失败")
            print("error type:", type(e).__name__)
            print("error message:", str(e))
            traceback.print_exc()

except Exception as e:
    print("transformers 无法导入")
    print("error type:", type(e).__name__)
    print("error message:", str(e))
    traceback.print_exc()


print_title("4. pytorch_model.bin 加载检查")

checkpoint = None

try:
    try:
        checkpoint = torch.load(
            WEIGHT_PATH,
            map_location="cpu",
            weights_only=True,
        )
        print("torch.load(weights_only=True) 成功")
    except TypeError:
        checkpoint = torch.load(
            WEIGHT_PATH,
            map_location="cpu",
        )
        print("当前 PyTorch 不支持 weights_only 参数，普通 torch.load 成功")
    except Exception as first_error:
        print("weights_only=True 加载失败：", repr(first_error))
        print("尝试普通 torch.load")

        checkpoint = torch.load(
            WEIGHT_PATH,
            map_location="cpu",
            weights_only=False,
        )
        print("torch.load(weights_only=False) 成功")

except Exception as e:
    print("pytorch_model.bin 加载失败")
    print("error type:", type(e).__name__)
    print("error message:", str(e))
    traceback.print_exc()


if checkpoint is not None:
    print("\ncheckpoint Python type:", type(checkpoint))

    if isinstance(checkpoint, torch.nn.Module):
        print("权重文件中保存的是完整 torch.nn.Module")
        print("model class:", checkpoint.__class__.__name__)
        print(
            "parameter count:",
            sum(p.numel() for p in checkpoint.parameters()),
        )

    elif isinstance(checkpoint, dict):
        print("checkpoint 顶层键数量:", len(checkpoint))
        print("checkpoint 顶层键:", list(checkpoint.keys())[:50])

        candidate_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
        ]

        state_dict = None

        for key in candidate_keys:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                print(f"发现候选参数字典: checkpoint['{key}']")
                state_dict = value
                break

        if state_dict is None:
            tensor_ratio = sum(
                isinstance(v, torch.Tensor)
                for v in checkpoint.values()
            )

            if tensor_ratio > 0:
                print("顶层字典本身看起来就是 state_dict")
                state_dict = checkpoint

        if state_dict is not None:
            print("\nstate_dict 参数项数量:", len(state_dict))

            total_values = 0
            tensor_count = 0

            for index, (key, value) in enumerate(state_dict.items()):
                if isinstance(value, torch.Tensor):
                    tensor_count += 1
                    total_values += value.numel()

                    if index < 100:
                        print(
                            f"{index:03d} | "
                            f"{key:70s} | "
                            f"shape={tuple(value.shape)} | "
                            f"dtype={value.dtype}"
                        )
                else:
                    if index < 100:
                        print(
                            f"{index:03d} | "
                            f"{key:70s} | "
                            f"type={type(value)}"
                        )

            print("\ntensor count:", tensor_count)
            print("total tensor values:", total_values)
            print(
                "estimated FP32 parameter size MB:",
                total_values * 4 / 1024 / 1024,
            )
        else:
            print("没有识别出标准 state_dict")

    else:
        print("该 checkpoint 既不是 nn.Module，也不是标准字典")
        print("checkpoint repr:", repr(checkpoint)[:2000])


print_title("5. 初步检查结束")
