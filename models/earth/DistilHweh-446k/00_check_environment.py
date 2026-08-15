\
import sys
import platform

print("=" * 70)
print("DistilHweh-446k environment check")
print("=" * 70)
print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())

mods = ["torch", "transformers", "safetensors", "huggingface_hub", "numpy"]
for name in mods:
    try:
        m = __import__(name)
        print(f"{name}: {getattr(m, '__version__', 'version unknown')}")
    except Exception as e:
        print(f"{name}: NOT AVAILABLE -> {type(e).__name__}: {e}")

try:
    import torch
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}:", torch.cuda.get_device_name(i))
    else:
        print("CUDA is not available. This is OK for this 446k-parameter smoke test; CPU can run it.")
except Exception:
    pass
