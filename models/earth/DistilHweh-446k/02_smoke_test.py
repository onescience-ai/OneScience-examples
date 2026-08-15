\
from __future__ import annotations

from pathlib import Path
import inspect
import importlib
import json
import sys
import traceback

import torch
import transformers
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parent
REQUIRED = ["config.json", "configuration.py", "modeling.py", "model.safetensors"]

print("=" * 78)
print("DistilHweh-446k local smoke test")
print("=" * 78)
print("Model directory:", ROOT)

missing = [name for name in REQUIRED if not (ROOT / name).exists()]
if missing:
    raise SystemExit(
        "Missing original Hugging Face model files: "
        + ", ".join(missing)
        + "\nPut the original repository files in this same directory first."
    )

raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
print("\nconfig.json keys:")
print(sorted(raw.keys()))
print("\nauto_map:")
print(json.dumps(raw.get("auto_map", {}), ensure_ascii=False, indent=2))

cfg = AutoConfig.from_pretrained(
    str(ROOT),
    trust_remote_code=True,
    local_files_only=True,
)
print("\nLoaded config class:", type(cfg).__name__)

# Try the Auto* class explicitly declared by config.json.
auto_map = raw.get("auto_map", {}) or {}
model = None
load_errors = []

preferred_auto_keys = [
    "AutoModel",
    "AutoModelForSequenceClassification",
    "AutoModelForRegression",
    "AutoModelForTimeSeriesPrediction",
    "AutoModelForTimeSeriesClassification",
]

candidate_auto_keys = preferred_auto_keys + [
    k for k in auto_map.keys() if k not in preferred_auto_keys and k != "AutoConfig"
]

for key in candidate_auto_keys:
    if key not in auto_map:
        continue
    cls = getattr(transformers, key, None)
    if cls is None:
        continue
    try:
        print(f"\nTrying {key}.from_pretrained(...)")
        model = cls.from_pretrained(
            str(ROOT),
            config=cfg,
            trust_remote_code=True,
            local_files_only=True,
        )
        print("Loaded with:", key)
        break
    except Exception as e:
        load_errors.append((key, repr(e)))

# Common fallback.
if model is None:
    try:
        from transformers import AutoModel
        print("\nTrying generic AutoModel.from_pretrained(...)")
        model = AutoModel.from_pretrained(
            str(ROOT),
            config=cfg,
            trust_remote_code=True,
            local_files_only=True,
        )
        print("Loaded with: AutoModel")
    except Exception as e:
        load_errors.append(("AutoModel", repr(e)))

# Manual fallback based on auto_map, useful if the repo registered only a custom class.
if model is None:
    mapping = None
    for key, value in auto_map.items():
        if key == "AutoConfig":
            continue
        mapping = value[0] if isinstance(value, (list, tuple)) else value
        if isinstance(mapping, str) and "." in mapping:
            break
        mapping = None

    if mapping:
        module_name, class_name = mapping.rsplit(".", 1)
        sys.path.insert(0, str(ROOT.parent))
        package_name = ROOT.name
        try:
            print(f"\nTrying manual class import: {package_name}.{module_name}.{class_name}")
            mod = importlib.import_module(f"{package_name}.{module_name}")
            model_cls = getattr(mod, class_name)
            model = model_cls.from_pretrained(str(ROOT), config=cfg, local_files_only=True)
            print("Loaded with manual custom class:", class_name)
        except Exception as e:
            load_errors.append(("manual import", repr(e)))

if model is None:
    print("\nMODEL LOAD FAILED. Attempts:")
    for name, err in load_errors:
        print(" -", name, "->", err)
    raise SystemExit(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
print("\nDevice:", device)
print("Model class:", type(model).__name__)
print("Parameter count:", f"{sum(p.numel() for p in model.parameters()):,}")

seq_len = int(getattr(cfg, "seq_len", raw.get("seq_len", 72)))
input_dim = int(getattr(cfg, "input_dim", raw.get("input_dim", 22)))
num_locations = int(getattr(cfg, "num_locations", raw.get("num_locations", 82)))

x = torch.zeros((1, seq_len, input_dim), dtype=torch.float32, device=device)
loc = torch.zeros((1,), dtype=torch.long, device=device)
attn = torch.ones((1, seq_len), dtype=torch.long, device=device)

sig = inspect.signature(model.forward)
print("\nforward signature:", sig)
print("Synthetic feature tensor:", tuple(x.shape))
print("Synthetic location tensor:", tuple(loc.shape))

def build_signature_kwargs():
    kwargs = {}
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        n = p.name.lower()

        if n in {"return_dict", "output_attentions", "output_hidden_states"}:
            continue
        if "label" in n or "target" in n:
            continue
        if "location" in n or n in {"loc", "loc_id", "loc_ids"}:
            kwargs[p.name] = loc
            continue
        if "attention_mask" in n:
            kwargs[p.name] = attn
            continue
        if (
            n in {
                "x", "input", "inputs", "features", "input_features",
                "weather_features", "input_values", "sequence", "sequences",
            }
            or ("feature" in n and "label" not in n)
        ):
            kwargs[p.name] = x
            continue
    return kwargs

attempts = []
sig_kwargs = build_signature_kwargs()
if sig_kwargs:
    attempts.append(("signature-derived kwargs", lambda: model(**sig_kwargs)))

attempts.extend([
    ("model(x, loc)", lambda: model(x, loc)),
    ("model(x)", lambda: model(x)),
    ("input_features + location_ids",
     lambda: model(input_features=x, location_ids=loc)),
    ("inputs + location_ids",
     lambda: model(inputs=x, location_ids=loc)),
    ("x + location_ids",
     lambda: model(x=x, location_ids=loc)),
    ("weather_features + location_ids",
     lambda: model(weather_features=x, location_ids=loc)),
])

output = None
errors = []
with torch.no_grad():
    for name, fn in attempts:
        try:
            print(f"\nForward attempt: {name}")
            output = fn()
            print("SUCCESS")
            break
        except Exception as e:
            errors.append((name, f"{type(e).__name__}: {e}"))
            print("  failed:", type(e).__name__, e)

if output is None:
    print("\nAll forward-call patterns failed.")
    print("This usually means the model uses a less common argument name or expects extra metadata.")
    print("Send the terminal output (especially forward signature and config keys) back for adjustment.")
    print("\nAttempts:")
    for name, err in errors:
        print(" -", name, "->", err)
    raise SystemExit(3)

def describe(obj, prefix="output"):
    if torch.is_tensor(obj):
        print(f"{prefix}: Tensor shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device}")
        return
    if isinstance(obj, dict):
        print(f"{prefix}: dict keys={list(obj.keys())}")
        for k, v in obj.items():
            describe(v, f"{prefix}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        print(f"{prefix}: {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj):
            describe(v, f"{prefix}[{i}]")
        return
    # Hugging Face ModelOutput-like object
    if hasattr(obj, "keys") and callable(obj.keys):
        try:
            keys = list(obj.keys())
            print(f"{prefix}: {type(obj).__name__} keys={keys}")
            for k in keys:
                describe(getattr(obj, k), f"{prefix}.{k}")
            return
        except Exception:
            pass
    print(f"{prefix}: {type(obj).__name__} -> {repr(obj)[:500]}")

print("\nModel output structure:")
describe(output)

print("\n" + "=" * 78)
print("SMOKE TEST PASSED")
print("The pretrained weights were loaded and one forward pass completed successfully.")
print("This uses synthetic normalized inputs; it validates execution, not real-world forecast accuracy.")
print("=" * 78)
