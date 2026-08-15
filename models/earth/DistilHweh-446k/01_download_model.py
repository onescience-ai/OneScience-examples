\
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "Harley-ml/DistilHweh-446k"
HERE = Path(__file__).resolve().parent

print(f"Downloading {REPO_ID}")
print(f"Target directory: {HERE}")

snapshot_download(
    repo_id=REPO_ID,
    local_dir=str(HERE),
    allow_patterns=[
        "config.json",
        "configuration.py",
        "modeling.py",
        "model.safetensors",
        "__init__.py",
        "README.md",
        ".gitattributes",
    ],
)

print("\nDownload finished.")
print("Files now present:")
for p in sorted(HERE.iterdir()):
    if p.is_file():
        print(f"  {p.name:24s} {p.stat().st_size / 1024:.1f} KiB")
