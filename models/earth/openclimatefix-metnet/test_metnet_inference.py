import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT
MODEL_DIR = ROOT / "hf_snapshot"
OUTPUT_DIR = ROOT / "outputs"

CONFIG_PATH = MODEL_DIR / "config.json"
WEIGHT_PATH = MODEL_DIR / "pytorch_model.bin"

sys.path.insert(0, str(SOURCE_DIR))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def create_synthetic_input(
    batch_size,
    sequence_length,
    channels,
    height,
    width,
):
    """
    构造确定性的虚拟气象图像序列。

    输入格式：
        [batch, time, channel, height, width]

    数据由空间梯度、周期信号和移动高斯场组成，
    数值范围控制在约 0 到 1。
    """
    torch.manual_seed(42)

    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    output = torch.empty(
        batch_size,
        sequence_length,
        channels,
        height,
        width,
        dtype=torch.float32,
    )

    for b in range(batch_size):
        for t in range(sequence_length):
            center_x = -0.35 + 0.25 * t
            center_y = 0.20 - 0.15 * t

            gaussian = torch.exp(
                -(
                    (xx - center_x) ** 2
                    + (yy - center_y) ** 2
                )
                / 0.18
            )

            for c in range(channels):
                phase = (c + 1) * 0.15
                periodic = torch.sin(
                    (c % 4 + 1) * torch.pi * xx + phase
                ) * torch.cos(
                    (c % 3 + 1) * torch.pi * yy - phase
                )

                field = (
                    0.45
                    + 0.20 * gaussian
                    + 0.15 * periodic
                    + 0.10 * xx
                    + 0.05 * yy
                )

                output[b, t, c] = field.clamp(0.0, 1.0)

    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "model": "openclimatefix/metnet",
        "status": "FAILED",
        "stage": "initialization",
    }

    try:
        from metnet import MetNet

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

        print("=" * 100)
        print("MetNet inference test")
        print("=" * 100)
        print("config:")
        print(json.dumps(config, indent=2, ensure_ascii=False))

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("python:", platform.python_version())
        print("torch:", torch.__version__)
        print("device:", device)

        if device.type == "cuda":
            print("accelerator:", torch.cuda.get_device_name(0))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        else:
            print("accelerator: CPU")

        print("\nConstructing official MetNet model...")

        model = MetNet(config=config)

        checkpoint = load_checkpoint(WEIGHT_PATH)

        print("checkpoint type:", type(checkpoint))
        print("checkpoint entries:", len(checkpoint))

        incompatible = model.load_state_dict(
            checkpoint,
            strict=True,
        )

        print("strict weight loading: SUCCESS")
        print("missing keys:", list(incompatible.missing_keys))
        print("unexpected keys:", list(incompatible.unexpected_keys))

        trainable_parameters = sum(
            p.numel() for p in model.parameters()
        )
        total_state_values = sum(
            value.numel()
            for value in model.state_dict().values()
            if isinstance(value, torch.Tensor)
        )

        batch_size = 1
        sequence_length = 2
        input_channels = int(config["input_channels"])
        input_size = int(config["input_size"])
        original_size = input_size * 4

        synthetic_input = create_synthetic_input(
            batch_size=batch_size,
            sequence_length=sequence_length,
            channels=input_channels,
            height=original_size,
            width=original_size,
        )

        print("\ninput shape:", tuple(synthetic_input.shape))
        print(
            "input min/max/mean/std:",
            float(synthetic_input.min()),
            float(synthetic_input.max()),
            float(synthetic_input.mean()),
            float(synthetic_input.std()),
        )

        model = model.to(device)
        model.eval()
        synthetic_input = synthetic_input.to(device)

        lead_time = 0

        synchronize(device)
        start = time.perf_counter()

        with torch.inference_mode():
            prediction = model(
                synthetic_input,
                lead_time=lead_time,
            )

        synchronize(device)
        elapsed_seconds = time.perf_counter() - start

        prediction_cpu = prediction.detach().float().cpu()
        input_cpu = synthetic_input.detach().float().cpu()

        peak_memory_bytes = (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0
        )

        print("\nforward status: SUCCESS")
        print("lead time:", lead_time)
        print("output shape:", tuple(prediction_cpu.shape))
        print(
            "output min/max/mean/std:",
            float(prediction_cpu.min()),
            float(prediction_cpu.max()),
            float(prediction_cpu.mean()),
            float(prediction_cpu.std()),
        )
        print("inference time seconds:", elapsed_seconds)
        print(
            "peak GPU memory GB:",
            peak_memory_bytes / 1024**3,
        )
        print("trainable parameters:", trainable_parameters)
        print("state tensor values:", total_state_values)

        np.savez_compressed(
            OUTPUT_DIR / "metnet_synthetic_test_output.npz",
            input=input_cpu.numpy(),
            prediction=prediction_cpu.numpy(),
            lead_time=np.array(lead_time),
        )

        result.update(
            {
                "status": "PASS",
                "stage": "completed",
                "device": str(device),
                "accelerator": (
                    torch.cuda.get_device_name(0)
                    if device.type == "cuda"
                    else "CPU"
                ),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "config": config,
                "strict_weight_loading": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "trainable_parameters": trainable_parameters,
                "state_tensor_values": total_state_values,
                "input_shape": list(input_cpu.shape),
                "input_min": float(input_cpu.min()),
                "input_max": float(input_cpu.max()),
                "input_mean": float(input_cpu.mean()),
                "input_std": float(input_cpu.std()),
                "lead_time": lead_time,
                "output_shape": list(prediction_cpu.shape),
                "output_min": float(prediction_cpu.min()),
                "output_max": float(prediction_cpu.max()),
                "output_mean": float(prediction_cpu.mean()),
                "output_std": float(prediction_cpu.std()),
                "inference_time_seconds": elapsed_seconds,
                "peak_gpu_memory_bytes": peak_memory_bytes,
                "peak_gpu_memory_gb": (
                    peak_memory_bytes / 1024**3
                ),
            }
        )

    except Exception as e:
        result["status"] = "FAILED"
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()

        print("\nforward status: FAILED")
        print("error type:", type(e).__name__)
        print("error message:", str(e))
        traceback.print_exc()

    result_path = OUTPUT_DIR / "metnet_test_result.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nresult saved:", result_path)

    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
