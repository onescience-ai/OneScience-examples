#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://huggingface.co/stockeh/swift-era5-1.4/resolve/main"

FILES=(
  "weights/swift/020000/.hydra/config.yaml"
  "weights/swift/020000/checkpoints/checkpoint-020000.pt"
  "sample-data/normalize_mean.npz"
  "sample-data/normalize_std.npz"
  "sample-data/normalize_diff_std_6.npz"
  "sample-data/test/2020_0937.h5"
  "sample-data/test/2020_0938.h5"
)

for relative_path in "${FILES[@]}"; do
    destination="${ROOT_DIR}/${relative_path}"
    temporary="${destination}.part"

    mkdir -p "$(dirname "${destination}")"

    if [[ -s "${destination}" ]]; then
        echo "[跳过] ${relative_path}"
        continue
    fi

    echo "[下载] ${relative_path}"

    curl \
      --location \
      --fail \
      --show-error \
      --retry 5 \
      --retry-delay 3 \
      --continue-at - \
      --output "${temporary}" \
      "${BASE_URL}/${relative_path}?download=true"

    mv -f "${temporary}" "${destination}"

    echo "[完成] ${relative_path}"
done

echo
echo "Swift 权重和样例数据下载完成。"
