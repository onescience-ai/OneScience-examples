"""Download and extract the Jena Climate dataset."""

from pathlib import Path
from urllib.request import urlretrieve
import zipfile


DATA_URL = (
    "https://storage.googleapis.com/tensorflow/"
    "tf-keras-datasets/jena_climate_2009_2016.csv.zip"
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
ZIP_PATH = DATA_DIR / "jena_climate_2009_2016.csv.zip"
PART_PATH = DATA_DIR / "jena_climate_2009_2016.csv.zip.part"
CSV_PATH = DATA_DIR / "jena_climate_2009_2016.csv"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 10 * 1024 * 1024:
        size_mb = CSV_PATH.stat().st_size / 1024 / 1024
        print(f"数据已经存在：{CSV_PATH}")
        print(f"CSV大小：{size_mb:.2f} MiB")
        return

    print(f"正在下载：{DATA_URL}")
    urlretrieve(DATA_URL, PART_PATH)
    PART_PATH.replace(ZIP_PATH)

    if not zipfile.is_zipfile(ZIP_PATH):
        raise RuntimeError(f"下载文件不是有效ZIP压缩包：{ZIP_PATH}")

    print("正在解压数据……")
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(DATA_DIR)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"解压后没有找到：{CSV_PATH}")

    size_mb = CSV_PATH.stat().st_size / 1024 / 1024
    print(f"数据准备完成：{CSV_PATH}")
    print(f"CSV大小：{size_mb:.2f} MiB")

    ZIP_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
