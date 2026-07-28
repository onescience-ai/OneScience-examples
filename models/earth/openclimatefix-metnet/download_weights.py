from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "openclimatefix/metnet"
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "hf_snapshot"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        "config.json",
        "pytorch_model.bin",
    ]

    print("=" * 80)
    print("Downloading MetNet files from Hugging Face")
    print("Repository:", REPO_ID)
    print("Target:", MODEL_DIR)
    print("=" * 80)

    for filename in files:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=str(MODEL_DIR),
        )
        print(f"Downloaded: {filename}")
        print(f"Saved to: {path}")

    print("=" * 80)
    print("Download completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
