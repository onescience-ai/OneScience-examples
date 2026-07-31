"""Download weights and sample data for Diffusion_SolRad."""

from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "thingnario/Diffusion_SolRad"
REVISION = "f11f15efba55638bf839e205e34d1f8e75a4e5b1"
ROOT_DIR = Path(__file__).resolve().parent

FILES = (
    "model_weights/ft06_01hr/weights.ckpt",
    "sample_data/sample_202504131100.npz",
    "sample_data/sample_202504161200.npz",
    "sample_data/sample_202507151200.npz",
)


def main() -> None:
    print("=" * 72)
    print("Diffusion_SolRad 资源下载")
    print("Repository:", REPO_ID)
    print("Revision:", REVISION)
    print("Target:", ROOT_DIR)
    print("=" * 72)

    for index, filename in enumerate(FILES, start=1):
        print(f"\n[{index}/{len(FILES)}] 下载：{filename}")

        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            revision=REVISION,
            local_dir=str(ROOT_DIR),
        )

        print("保存位置：", downloaded_path)

    revision_path = ROOT_DIR / "results" / "huggingface_revision.txt"
    revision_path.parent.mkdir(parents=True, exist_ok=True)
    revision_path.write_text(f"{REVISION}\n", encoding="utf-8")

    print("\n资源下载完成")
    print("版本记录：", revision_path)


if __name__ == "__main__":
    main()
