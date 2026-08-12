#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: download_official_resources.sh ROOT_DIR}"
COMMIT="055f858e68f5e0151eb6fece9f6b574d3da4af8d"
REPOSITORY="https://github.com/clessig/atmorep.git"
MODEL_URL="https://datapub.fz-juelich.de/atmorep/models/model_id4nvwbetz.tar.gz"

mkdir -p "$ROOT/vendor" "$ROOT/resources"
if [[ ! -d "$ROOT/vendor/atmorep-official/.git" ]]; then
  git clone --no-checkout "$REPOSITORY" "$ROOT/vendor/atmorep-official"
fi
git -C "$ROOT/vendor/atmorep-official" fetch origin "$COMMIT"
git -C "$ROOT/vendor/atmorep-official" checkout --detach "$COMMIT"
curl -fL --retry 3 --continue-at - -o "$ROOT/resources/model_id4nvwbetz.tar.gz" "$MODEL_URL"
gzip -t "$ROOT/resources/model_id4nvwbetz.tar.gz"
tar -xzf "$ROOT/resources/model_id4nvwbetz.tar.gz" -C "$ROOT/resources"
sha256sum "$ROOT/resources/model_id4nvwbetz.tar.gz" \
  "$ROOT/resources/id4nvwbetz/AtmoRep_id4nvwbetz.mod" \
  "$ROOT/resources/id4nvwbetz/model_id4nvwbetz.json"
