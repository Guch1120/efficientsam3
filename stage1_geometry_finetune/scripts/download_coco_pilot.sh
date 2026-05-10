#!/usr/bin/env bash
set -euo pipefail

# COCO pilot 用の最小構成を取得する。
# raw COCO の標準配置をそのまま保持する。

DATA_ROOT="${DATA_ROOT:-/ros2_ws/efficientsam3/data/coco_pilot}"
TMP_DIR="${TMP_DIR:-${DATA_ROOT}/_downloads}"
DOWNLOAD_TRAIN="${DOWNLOAD_TRAIN:-1}"
DOWNLOAD_VAL="${DOWNLOAD_VAL:-1}"

mkdir -p "${DATA_ROOT}" "${TMP_DIR}"

download_if_missing() {
  local url="$1"
  local out="$2"
  local partial="${out}.part"
  if [ -f "${out}" ]; then
    if unzip -tqq "${out}" >/dev/null 2>&1; then
      echo "skip existing: ${out}"
      return 0
    fi
    echo "remove broken archive: ${out}"
    rm -f "${out}"
  fi
  if [ -f "${partial}" ]; then
    echo "resume partial: ${partial}"
    wget -c -nv -O "${partial}" "${url}"
    mv "${partial}" "${out}"
    return 0
  fi
  echo "download: ${url}"
  wget -c -nv -O "${partial}" "${url}"
  mv "${partial}" "${out}"
}

extract_if_missing() {
  local zip_path="$1"
  local target_dir="$2"
  local marker="$3"
  if [ -e "${marker}" ]; then
    echo "skip extracted: ${marker}"
    return 0
  fi
  mkdir -p "${target_dir}"
  echo "extract: ${zip_path}"
  unzip -q "${zip_path}" -d "${target_dir}"
}

download_if_missing \
  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
  "${TMP_DIR}/annotations_trainval2017.zip"
if [ "${DOWNLOAD_TRAIN}" = "1" ]; then
  download_if_missing \
    "http://images.cocodataset.org/zips/train2017.zip" \
    "${TMP_DIR}/train2017.zip"
fi
if [ "${DOWNLOAD_VAL}" = "1" ]; then
  download_if_missing \
    "http://images.cocodataset.org/zips/val2017.zip" \
    "${TMP_DIR}/val2017.zip"
fi

extract_if_missing \
  "${TMP_DIR}/annotations_trainval2017.zip" \
  "${DATA_ROOT}" \
  "${DATA_ROOT}/annotations/instances_train2017.json"
if [ "${DOWNLOAD_TRAIN}" = "1" ]; then
  extract_if_missing \
    "${TMP_DIR}/train2017.zip" \
    "${DATA_ROOT}" \
    "${DATA_ROOT}/train2017"
fi
if [ "${DOWNLOAD_VAL}" = "1" ]; then
  extract_if_missing \
    "${TMP_DIR}/val2017.zip" \
    "${DATA_ROOT}" \
    "${DATA_ROOT}/val2017"
fi

echo "COCO pilot root: ${DATA_ROOT}"
du -sh "${DATA_ROOT}" || true
find "${DATA_ROOT}/annotations" -maxdepth 1 -type f | sort || true
