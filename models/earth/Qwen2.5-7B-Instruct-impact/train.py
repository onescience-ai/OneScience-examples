#!/usr/bin/env python3
"""Functional validation for Qwen2.5-7B-Instruct-impact.

The script follows the official model card's vLLM inference path, uses the
repository's own Qwen 2.5 chat template, and evaluates strict Yes/No impact
classification. If no labelled CSV/JSON/JSONL file is present, a deterministic
small synthetic disclosure set is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MODEL_ID = "extreme-weather-impacts/Qwen2.5-7B-Instruct-impact"
MODEL_REVISION = "4aa57015775cd4feac38a2c9b564430d5dbff53e"

WEIGHT_FILES: Dict[str, Dict[str, Any]] = {
    "model-00001-of-00004.safetensors": {
        "bytes": 4_877_660_672,
        "sha256": "57d0f67fd5f05652f97c51057a46d2cb4217d50615f6b1b2304c740ceb240d76",
    },
    "model-00002-of-00004.safetensors": {
        "bytes": 4_932_750_888,
        "sha256": "10e6d1e753d4fa4d729125b7235b0e307c24c62351aa89e7e6ede831898de989",
    },
    "model-00003-of-00004.safetensors": {
        "bytes": 4_330_865_088,
        "sha256": "b5f4e515d6b12510d85d11d60c436922b5d3f70bc8e1283fac098db7fc753a6b",
    },
    "model-00004-of-00004.safetensors": {
        "bytes": 1_089_994_880,
        "sha256": "053d06aa9a6579967b3653c789151a7edf361899e7804c10a756b53286f602e8",
    },
}

REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

PROMPT_TEMPLATE_IMPACT = """You are given a TEXT of a company disclosure. Your task is to determine whether the company was exposed to an extreme weather event based on the TEXT.

Here is the TEXT from the company’s disclosure:
[begin of TEXT]
{text}
[end of TEXT]
Answer the following questions strictly with "Yes" or "No":
- Based on the TEXT, was the company exposed to an extreme weather event (e.g., Storm, Flood, Heatwave, Drought, Wildfire, Coldwave)?
Decision Guidelines:
- A company is considered "exposed" only if:
  1. It was directly impacted by an extreme weather event mentioned in the TEXT.
  2. The impact happened in the past and is explicitly linked to the company.
  3. The impact was caused by a clear extreme weather event, not ordinary weather conditions.
- Forward-looking statements, potential future impacts, or potential risks do NOT count as "exposed".
- Merely stating a geographic location does NOT count as "exposed".
- Merely stating a generic or specific list of extreme weather events does NOT count as "exposed".
- TEXTs that are not full sentences do NOT count as "exposed".
Output Format:
Only respond by strictly giving a "Yes" or "No".

Your Output:
"""

SYNTHETIC_EXAMPLES: Tuple[Dict[str, str], ...] = (
    {
        "id": "syn-001",
        "text": "The most severe forward-looking risks for our firm are hurricanes and wildfires.",
        "label": "No",
    },
    {
        "id": "syn-002",
        "text": "Last year, a large freeze in Texas resulted in the closure of our production facilities.",
        "label": "Yes",
    },
    {
        "id": "syn-003",
        "text": "Flooding damaged our main warehouse in May and halted customer shipments for six days.",
        "label": "Yes",
    },
    {
        "id": "syn-004",
        "text": "We operate offices in Florida, California, and Texas.",
        "label": "No",
    },
    {
        "id": "syn-005",
        "text": "A wildfire destroyed inventory at our regional distribution center during the prior fiscal year.",
        "label": "Yes",
    },
    {
        "id": "syn-006",
        "text": "Climate-related risks may include storms, floods, droughts, and heatwaves.",
        "label": "No",
    },
    {
        "id": "syn-007",
        "text": "The July heatwave caused cooling equipment to fail and forced our plant to suspend operations.",
        "label": "Yes",
    },
    {
        "id": "syn-008",
        "text": "Severe weather could disrupt our suppliers and increase transportation costs in the future.",
        "label": "No",
    },
    {
        "id": "syn-009",
        "text": "A prolonged drought reduced water availability and lowered production at two of our sites last year.",
        "label": "Yes",
    },
    {
        "id": "syn-010",
        "text": "Our insurance policy covers losses caused by hurricanes, wildfires, and floods.",
        "label": "No",
    },
    {
        "id": "syn-011",
        "text": "Hurricane winds damaged our coastal facility in September and resulted in a three-week shutdown.",
        "label": "Yes",
    },
    {
        "id": "syn-012",
        "text": "Storm, flood, heatwave, drought, wildfire, coldwave.",
        "label": "No",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the official Qwen2.5 impact-classification model."
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--backend",
        default="vllm",
        choices=("vllm", "transformers"),
        help="vllm matches the official model card; transformers is a fallback.",
    )
    parser.add_argument(
        "--verify",
        default="full",
        choices=("full", "size", "none"),
        help="Verify all shard SHA256 hashes, sizes only, or skip verification.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def package_version(distribution_name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(distribution_name)
    except Exception:
        return None


def dependency_report(backend: str) -> Dict[str, Dict[str, Any]]:
    specs = [
        ("torch", "torch", True),
        ("transformers", "transformers", True),
        ("safetensors", "safetensors", True),
        ("vllm", "vllm", backend == "vllm"),
        ("matplotlib", "matplotlib", False),
    ]
    report: Dict[str, Dict[str, Any]] = {}
    for module_name, distribution_name, required in specs:
        try:
            module = importlib.import_module(module_name)
            report[module_name] = {
                "required": required,
                "import_ok": True,
                "version": package_version(distribution_name)
                or getattr(module, "__version__", "unknown"),
                "error": None,
            }
        except Exception as error:
            report[module_name] = {
                "required": required,
                "import_ok": False,
                "version": package_version(distribution_name),
                "error": f"{type(error).__name__}: {error}",
            }
    return report


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_repository(model_dir: Path, mode: str) -> Dict[str, Any]:
    missing = [
        name
        for name in (*REQUIRED_FILES, *WEIGHT_FILES.keys())
        if not (model_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing official model files: " + ", ".join(missing)
        )

    shard_results: Dict[str, Any] = {}
    for name, expected in WEIGHT_FILES.items():
        path = model_dir / name
        size = path.stat().st_size
        size_ok = size == expected["bytes"]
        result: Dict[str, Any] = {
            "bytes": size,
            "expected_bytes": expected["bytes"],
            "size_verified": size_ok,
            "sha256": None,
            "expected_sha256": expected["sha256"],
            "sha256_verified": None,
        }
        if mode == "full":
            print(f"SHA256 {name} ({size / 1e9:.2f} GB)...", flush=True)
            actual_hash = sha256_file(path)
            result["sha256"] = actual_hash
            result["sha256_verified"] = actual_hash == expected["sha256"]
        shard_results[name] = result

    if mode != "none":
        bad_sizes = [
            name for name, result in shard_results.items()
            if not result["size_verified"]
        ]
        bad_hashes = [
            name for name, result in shard_results.items()
            if result["sha256_verified"] is False
        ]
        if bad_sizes or bad_hashes:
            raise RuntimeError(
                "Official shard verification failed. "
                f"Bad sizes={bad_sizes}; bad hashes={bad_hashes}"
            )

    return {
        "mode": mode,
        "revision": MODEL_REVISION,
        "total_weight_bytes": sum(
            result["bytes"] for result in shard_results.values()
        ),
        "all_sizes_verified": all(
            result["size_verified"] for result in shard_results.values()
        ),
        "all_sha256_verified": (
            all(result["sha256_verified"] for result in shard_results.values())
            if mode == "full"
            else None
        ),
        "shards": shard_results,
    }


def count_safetensors_parameters(model_dir: Path) -> int:
    """Count tensor elements from safetensors headers without loading weights."""
    total = 0
    seen = set()
    for filename in WEIGHT_FILES:
        with (model_dir / filename).open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), byteorder="little")
            header = json.loads(handle.read(header_size).decode("utf-8"))
        for tensor_name, tensor_info in header.items():
            if tensor_name == "__metadata__" or tensor_name in seen:
                continue
            count = math.prod(int(value) for value in tensor_info["shape"])
            total += count
            seen.add(tensor_name)
    return total


def normalize_label(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)) and value in (0, 1):
        return "Yes" if int(value) == 1 else "No"
    text = str(value).strip().lower()
    yes_values = {"yes", "y", "1", "true", "impact", "impacted", "exposed"}
    no_values = {"no", "n", "0", "false", "no impact", "not exposed"}
    if text in yes_values:
        return "Yes"
    if text in no_values:
        return "No"
    return None


def records_from_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    text_columns = ("text", "content", "disclosure", "sentence", "input")
    label_columns = ("label", "target", "impact", "exposed", "answer")
    records: List[Dict[str, str]] = []
    for index, row in enumerate(rows):
        text = next(
            (str(row[key]).strip() for key in text_columns if row.get(key)),
            "",
        )
        raw_label = next(
            (row[key] for key in label_columns if key in row),
            None,
        )
        label = normalize_label(raw_label)
        if text and label:
            records.append(
                {
                    "id": str(row.get("id") or f"local-{index + 1:04d}"),
                    "text": text,
                    "label": label,
                }
            )
    return records


def load_dataset_file(path: Path) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return records_from_rows(csv.DictReader(handle))
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            return records_from_rows(
                json.loads(line) for line in handle if line.strip()
            )
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            for key in ("data", "records", "examples", "test"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            return []
        return records_from_rows(payload)
    return []


def find_or_create_test_data(
    model_dir: Path,
    explicit_path: Optional[Path],
) -> Tuple[List[Dict[str, str]], str, Optional[str]]:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        records = load_dataset_file(path)
        if not records:
            raise ValueError(
                f"No usable text/label records found in explicit data file: {path}"
            )
        return records, "local_labelled_file", str(path)

    excluded_names = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "model_manifest.json",
        "metrics.json",
    }
    candidates: List[Path] = []
    for pattern in ("*.csv", "*.jsonl", "*.json"):
        candidates.extend(model_dir.glob(pattern))
    for path in sorted(candidates):
        if path.name in excluded_names or "validation_results" in path.parts:
            continue
        records = load_dataset_file(path)
        if records:
            return records, "local_labelled_file", str(path)

    return [dict(item) for item in SYNTHETIC_EXAMPLES], "synthetic_disclosures", None


def build_prompts(tokenizer: Any, records: Sequence[Dict[str, str]]) -> List[str]:
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "The local tokenizer has no chat_template. Re-download "
            "tokenizer_config.json from the official repository."
        )
    prompts = []
    for record in records:
        content = PROMPT_TEMPLATE_IMPACT.format(text=record["text"])
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def generate_vllm(
    model_dir: Path,
    prompts: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[str], List[float]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        dtype="float16",
        trust_remote_code=False,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=0.01,
        min_p=0.1,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    generated: List[str] = []
    per_item_latencies: List[float] = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        began = time.perf_counter()
        outputs = llm.generate(batch, sampling_params, use_tqdm=False)
        elapsed = time.perf_counter() - began
        generated.extend(item.outputs[0].text for item in outputs)
        per_item_latencies.extend([elapsed / len(batch)] * len(batch))
    return generated, per_item_latencies


def generate_transformers(
    model_dir: Path,
    tokenizer: Any,
    prompts: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[str], List[float]]:
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    load_kwargs: Dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": dtype,
        "trust_remote_code": False,
    }
    if args.device == "cuda":
        load_kwargs["device_map"] = {"": "cuda"}
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    if args.device == "cpu":
        model.to("cpu")
    model.eval()

    tokenizer.padding_side = "left"
    generated: List[str] = []
    per_item_latencies: List[float] = []
    for start in range(0, len(prompts), args.batch_size):
        batch = list(prompts[start : start + args.batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_model_len - args.max_new_tokens,
        )
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        began = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        prompt_width = encoded["input_ids"].shape[1]
        completions = output_ids[:, prompt_width:]
        generated.extend(
            tokenizer.batch_decode(completions, skip_special_tokens=True)
        )
        per_item_latencies.extend([elapsed / len(batch)] * len(batch))
    return generated, per_item_latencies


def parse_answer(text: str) -> Tuple[Optional[str], bool]:
    stripped = text.strip()
    strict = re.fullmatch(r"(?i)(yes|no)[.!]?", stripped) is not None
    match = re.search(r"(?i)\b(yes|no)\b", stripped)
    if match is None:
        return None, strict
    return match.group(1).capitalize(), strict


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(
    records: Sequence[Dict[str, str]],
    completions: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    predictions: List[Dict[str, Any]] = []
    tp = tn = fp = fn = 0
    strict_count = parsed_count = correct_count = 0
    for record, completion in zip(records, completions):
        predicted, strict = parse_answer(completion)
        actual = record["label"]
        parsed_count += int(predicted is not None)
        strict_count += int(strict)
        correct = predicted == actual
        correct_count += int(correct)
        if actual == "Yes" and predicted == "Yes":
            tp += 1
        elif actual == "No" and predicted == "No":
            tn += 1
        elif actual == "No" and predicted == "Yes":
            fp += 1
        elif actual == "Yes" and predicted == "No":
            fn += 1
        predictions.append(
            {
                **record,
                "raw_completion": completion,
                "prediction": predicted,
                "strict_format": strict,
                "correct": correct,
            }
        )

    yes_precision = safe_div(tp, tp + fp)
    yes_recall = safe_div(tp, tp + fn)
    yes_f1 = safe_div(2 * yes_precision * yes_recall, yes_precision + yes_recall)
    no_precision = safe_div(tn, tn + fn)
    no_recall = safe_div(tn, tn + fp)
    no_f1 = safe_div(2 * no_precision * no_recall, no_precision + no_recall)
    count = len(records)
    return {
        "count": count,
        "accuracy": safe_div(correct_count, count),
        "macro_f1": (yes_f1 + no_f1) / 2,
        "yes_precision": yes_precision,
        "yes_recall": yes_recall,
        "yes_f1": yes_f1,
        "no_precision": no_precision,
        "no_recall": no_recall,
        "no_f1": no_f1,
        "parsed_fraction": safe_div(parsed_count, count),
        "strict_format_fraction": safe_div(strict_count, count),
        "confusion_matrix": {
            "true_yes_pred_yes": tp,
            "true_yes_pred_no": fn,
            "true_no_pred_yes": fp,
            "true_no_pred_no": tn,
        },
    }, predictions


def save_preview(
    output_path: Path,
    metrics: Dict[str, Any],
    data_source: str,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    values = [
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["parsed_fraction"],
        metrics["strict_format_fraction"],
    ]
    labels = ["Accuracy", "Macro F1", "Parsed", "Strict format"]
    matrix = metrics["confusion_matrix"]
    cm = [
        [matrix["true_no_pred_no"], matrix["true_no_pred_yes"]],
        [matrix["true_yes_pred_no"], matrix["true_yes_pred_yes"]],
    ]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    bars = axes[0].bar(labels, values, color=["#2c7fb8", "#41b6c4", "#7fcdbb", "#a1d76a"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Functional classification metrics")
    axes[0].tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )

    image = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks([0, 1], labels=["Pred No", "Pred Yes"])
    axes[1].set_yticks([0, 1], labels=["True No", "True Yes"])
    axes[1].set_title("Confusion matrix")
    for row in range(2):
        for col in range(2):
            axes[1].text(col, row, str(cm[row][col]), ha="center", va="center")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    figure.suptitle(
        f"Qwen2.5-7B-Instruct-impact — {data_source}",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def write_outputs(
    output_dir: Path,
    report: Dict[str, Any],
    predictions: Sequence[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "predictions.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "text",
                "label",
                "prediction",
                "strict_format",
                "correct",
                "raw_completion",
            ],
        )
        writer.writeheader()
        writer.writerows(predictions)

    classification = report["classification"]
    performance = report["performance"]
    summary = "\n".join(
        [
            "Qwen2.5-7B-Instruct-impact validation",
            "=" * 44,
            f"Status: {report['status']}",
            f"Validation level: {report['validation_level']}",
            f"Data source: {report['data_source']}",
            f"Backend: {report['environment']['backend']}",
            f"Device: {report['environment']['device']}",
            f"Repository revision: {report['repository']['revision']}",
            f"Weight verification: {report['repository']['verification']['mode']}",
            f"Parameters: {report['model']['parameter_count']:,}",
            f"Examples: {classification['count']}",
            f"Accuracy: {classification['accuracy']:.6f}",
            f"Macro F1: {classification['macro_f1']:.6f}",
            f"Parsed answers: {classification['parsed_fraction']:.6f}",
            f"Strict format: {classification['strict_format_fraction']:.6f}",
            f"Median latency: {performance['median_latency_seconds_per_example']:.3f} s/example",
            f"Peak GPU memory: {performance['peak_gpu_memory_mb']:.2f} MB",
            "Scientific accuracy: unavailable (synthetic examples only)"
            if report["data_source"] == "synthetic_disclosures"
            else "Scientific accuracy: local labelled file only; official benchmark not reproduced",
            f"Results: {output_dir.resolve()}",
            "",
        ]
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_dir = Path(__file__).resolve().parent
    output_dir = model_dir / "validation_results"

    print(f"Python:  {platform.python_version()}")
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception as error:
        print(f"PyTorch: import failed ({type(error).__name__}: {error})")
        cuda_available = False
        device_name = None
        torch = None  # type: ignore[assignment]
    print(f"Device:  {args.device}")

    dependencies = dependency_report(args.backend)
    required_failures = [
        name
        for name, item in dependencies.items()
        if item["required"] and not item["import_ok"]
    ]
    if args.device == "cuda" and not cuda_available:
        required_failures.append("CUDA")

    if args.preflight_only:
        print("\n========== Qwen impact model preflight ==========")
        print(
            "Status:             "
            + ("PASS" if not required_failures else "ENVIRONMENT_INCOMPLETE")
        )
        print(f"Model directory:    {model_dir}")
        print(f"Repository revision:{MODEL_REVISION}")
        missing_files = [
            name
            for name in (*REQUIRED_FILES, *WEIGHT_FILES.keys())
            if not (model_dir / name).is_file()
        ]
        print(
            "Official files:     "
            + ("PRESENT" if not missing_files else "MISSING")
        )
        if missing_files:
            print("Missing files:      " + ", ".join(missing_files))
        print("Dependency imports:")
        for name, item in dependencies.items():
            status = "OK" if item["import_ok"] else "MISSING/FAILED"
            required = "required" if item["required"] else "optional"
            print(
                f"  {name:<14} {status:<14} "
                f"{str(item['version'] or 'unknown'):<18} ({required})"
            )
            if item["error"]:
                print(f"    {item['error']}")
        if required_failures:
            print("Required failures:  " + ", ".join(required_failures))
        print("Inference:          not run (--preflight-only)")
        return 0 if not required_failures and not missing_files else 2

    if required_failures:
        raise RuntimeError(
            "Required runtime components failed: " + ", ".join(required_failures)
        )

    print("Verifying the fixed official repository files...")
    verification = verify_repository(model_dir, args.verify)
    print(
        "Repository verification: "
        + (
            "SHA256 VERIFIED"
            if verification["all_sha256_verified"]
            else "SIZES VERIFIED"
            if verification["all_sizes_verified"]
            else args.verify.upper()
        )
    )
    parameter_count = count_safetensors_parameters(model_dir)

    records, data_source, input_path = find_or_create_test_data(
        model_dir,
        args.data,
    )
    if args.limit is not None:
        records = records[: max(1, args.limit)]
    print(f"Data source: {data_source} ({len(records)} examples)")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    prompts = build_prompts(tokenizer, records)

    if torch is not None and args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"Loading the model with backend={args.backend}...")
    if args.backend == "vllm":
        completions, latencies = generate_vllm(model_dir, prompts, args)
    else:
        completions, latencies = generate_transformers(
            model_dir,
            tokenizer,
            prompts,
            args,
        )

    classification, predictions = calculate_metrics(records, completions)
    status = (
        "PASS"
        if len(completions) == len(records)
        and classification["parsed_fraction"] == 1.0
        else "FAIL"
    )
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch is not None and args.device == "cuda"
        else 0.0
    )
    report: Dict[str, Any] = {
        "status": status,
        "validation_level": "functional_only",
        "model_name": MODEL_ID,
        "data_source": data_source,
        "repository": {
            "revision": MODEL_REVISION,
            "model_directory": str(model_dir),
            "verification": verification,
        },
        "model": {
            "architecture": "Qwen2ForCausalLM",
            "parameter_count": parameter_count,
            "task": "binary extreme-weather impact detection",
            "prompt_output": "strict Yes/No",
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": getattr(torch, "__version__", None),
            "transformers": dependencies["transformers"]["version"],
            "vllm": dependencies["vllm"]["version"],
            "backend": args.backend,
            "device": args.device,
            "cuda_available": cuda_available,
            "cuda_device_name": device_name,
            "dependency_imports": dependencies,
        },
        "input": {
            "path": input_path,
            "example_count": len(records),
            "labels": ["No", "Yes"],
            "prompt_template": "official model-card impact prompt",
            "chat_template": "repository Qwen 2.5 chat template",
            "synthetic_examples": data_source == "synthetic_disclosures",
        },
        "generation": {
            "temperature": 0.01 if args.backend == "vllm" else 0.0,
            "min_p": 0.1 if args.backend == "vllm" else None,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
        },
        "classification": classification,
        "performance": {
            "latencies_seconds_per_example": latencies,
            "median_latency_seconds_per_example": statistics.median(latencies),
            "mean_latency_seconds_per_example": statistics.mean(latencies),
            "peak_gpu_memory_mb": peak_gpu_mb,
            "peak_memory_note": "PyTorch CUDA allocator-reported peak",
        },
        "metric_interpretation": (
            "Synthetic exact-match metrics validate the official prompt, "
            "tokenizer, generation and Yes/No evaluation path only. They do "
            "not reproduce the authors' held-out dataset evaluation."
        ),
        "official_reference": {
            "fine_tuning_data": "approximately 6k training examples",
            "dataset": "extreme-weather-impacts/impact_eventimpact",
            "official_backend": "vLLM",
            "license": "Apache-2.0",
        },
        "output_dir": str(output_dir.resolve()),
    }

    write_outputs(output_dir, report, predictions)
    preview_saved = save_preview(
        output_dir / "validation_preview.png",
        classification,
        data_source,
    )
    report["artifacts"] = {
        "metrics": str((output_dir / "metrics.json").resolve()),
        "summary": str((output_dir / "summary.txt").resolve()),
        "predictions": str((output_dir / "predictions.csv").resolve()),
        "preview": (
            str((output_dir / "validation_preview.png").resolve())
            if preview_saved
            else None
        ),
    }
    # Rewrite metrics once to include final artifact paths.
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== Qwen impact validation ==========")
    print(f"Status:              {status}")
    print("Validation level:    functional_only")
    print(f"Data source:         {data_source}")
    print(f"Backend:             {args.backend}")
    print(f"Device:              {args.device}")
    print(f"Repository revision: {MODEL_REVISION}")
    print(f"Weight verification: {args.verify}")
    print(f"Parameters:          {parameter_count:,}")
    print(f"Examples:            {classification['count']}")
    print(f"Accuracy:            {classification['accuracy']:.6f}")
    print(f"Macro F1:            {classification['macro_f1']:.6f}")
    print(f"Parsed answers:      {classification['parsed_fraction']:.6f}")
    print(
        f"Strict output format:{classification['strict_format_fraction']:.6f}"
    )
    print(
        "Median latency:      "
        f"{statistics.median(latencies):.3f} s/example"
    )
    print(f"Peak GPU memory:     {peak_gpu_mb:.2f} MB")
    if data_source == "synthetic_disclosures":
        print("Scientific accuracy: unavailable (synthetic examples only)")
    else:
        print("Scientific accuracy: local labelled file; official benchmark not reproduced")
    print(f"Results:             {output_dir.resolve()}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
