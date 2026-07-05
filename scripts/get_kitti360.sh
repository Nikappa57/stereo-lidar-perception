#!/usr/bin/env bash
#
# get_kitti360.sh — download + convert KITTI-360 (colour stereo + LiDAR + 3D boxes)
# into the py123d format used by the stereo-lidar-perception pipeline.
#
# What it does:
#   1. Downloads the PUBLIC per-sequence raw data from KITTI-360's S3 bucket
#      (rectified stereo image_00/image_01 + Velodyne scans + timestamps).
#   2. Extracts everything into the on-disk layout py123d expects.
#   3. Runs `py123d-conversion` into  $PY123D_DATA_ROOT/logs/kitti360_{train,val}/,
#      ready for the DataLoader / training.
#
# The small LOGIN-GATED files (calibration/, data_poses/, data_3d_bboxes/) are NOT
# public — download them once from https://www.cvlibs.net/datasets/kitti-360/ and
# drop them under $KITTI360_DATA_ROOT/. This script checks for them and stops with a
# clear message if they are missing.
#
# Usage:
#   scripts/get_kitti360.sh                       # smoke test: drive_0003 -> train
#   TRAIN_SEQ="0000 0002 0004 0005 0006 0007" \
#   VAL_SEQ="0009 0010" scripts/get_kitti360.sh   # a real train/val split
#   KEEP_ZIPS=1 scripts/get_kitti360.sh           # don't delete the downloaded zips
#   SKIP_CONVERT=1 scripts/get_kitti360.sh        # download+extract only
#
# Env overrides:
#   KITTI360_DATA_ROOT  (default: <repo>/KITTI-360)   raw dataset root
#   PY123D_DATA_ROOT    (default: <repo>/data)         converted-logs root
#   TRAIN_SEQ / VAL_SEQ  space-separated 4-digit sequence ids (see valid list below)
#
set -euo pipefail

# ----------------------------------------------------------------------------- paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KITTI360_DATA_ROOT="${KITTI360_DATA_ROOT:-$REPO_ROOT/KITTI-360}"
PY123D_DATA_ROOT="${PY123D_DATA_ROOT:-$REPO_ROOT/data}"
export KITTI360_DATA_ROOT PY123D_DATA_ROOT

S3="https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360"
ZIP_DIR="$KITTI360_DATA_ROOT/_zips"

# Sequences that KITTI-360 publishes raw stereo+velodyne for.
VALID_SEQ="0000 0002 0003 0004 0005 0006 0007 0009 0010"
TRAIN_SEQ="${TRAIN_SEQ:-0003}"
VAL_SEQ="${VAL_SEQ:-}"

seq_name() { echo "2013_05_28_drive_${1}_sync"; }

log()  { printf '\033[1;36m[kitti360]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[kitti360] ERROR:\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

# --------------------------------------------------------------- preconditions check
command -v wget  >/dev/null || die "wget not found"
command -v unzip >/dev/null || die "unzip not found"

for s in $TRAIN_SEQ $VAL_SEQ; do
  [[ " $VALID_SEQ " == *" $s "* ]] || die "invalid sequence '$s'. Valid: $VALID_SEQ"
done

# Login-gated bits must already be present (see header).
missing=()
[[ -f "$KITTI360_DATA_ROOT/calibration/perspective.txt"     ]] || missing+=("calibration/")
[[ -d "$KITTI360_DATA_ROOT/data_poses"                      ]] || missing+=("data_poses/")
[[ -d "$KITTI360_DATA_ROOT/data_3d_bboxes"                  ]] || missing+=("data_3d_bboxes/")
if (( ${#missing[@]} )); then
  err "Missing login-gated files under $KITTI360_DATA_ROOT: ${missing[*]}"
  err "Download them (small: ~3K / 8.9M / 30M) from the KITTI-360 download page:"
  err "  https://www.cvlibs.net/datasets/kitti-360/  ->  Calibrations, Vehicle Poses, 3D Bounding Boxes"
  err "and unzip them under \$KITTI360_DATA_ROOT, then re-run."
  exit 1
fi

mkdir -p "$ZIP_DIR" "$KITTI360_DATA_ROOT/data_2d_raw" "$KITTI360_DATA_ROOT/data_3d_raw"

# ---------------------------------------------------------------- download + extract
# $1 = relative S3 path, $2 = local zip name, $3 = extract dir, $4 = unzip filter (opt)
fetch_unzip() {
  local rel="$1" name="$2" dest="$3" filter="${4:-}"
  local zip="$ZIP_DIR/$name"
  if [[ ! -s "$zip" ]]; then
    log "downloading $name"
    wget -c -q --show-progress "$S3/$rel" -O "$zip" \
      || die "download failed: $S3/$rel"
  else
    log "already downloaded: $name"
  fi
  log "extracting $name -> ${dest#$KITTI360_DATA_ROOT/}"
  # shellcheck disable=SC2086
  unzip -o -q "$zip" $filter -d "$dest"
}

# Timestamps zips cover ALL sequences; fetch once, extract only the ones we need.
TS_PERSP="$ZIP_DIR/data_timestamps_perspective.zip"
TS_VELO="$ZIP_DIR/data_timestamps_velodyne.zip"
[[ -s "$TS_PERSP" ]] || { log "downloading data_timestamps_perspective.zip"; wget -c -q --show-progress "$S3/data_2d_raw/data_timestamps_perspective.zip" -O "$TS_PERSP"; }
[[ -s "$TS_VELO"  ]] || { log "downloading data_timestamps_velodyne.zip";  wget -c -q --show-progress "$S3/data_3d_raw/data_timestamps_velodyne.zip"  -O "$TS_VELO"; }

for s in $TRAIN_SEQ $VAL_SEQ; do
  seq="$(seq_name "$s")"
  log "=== sequence $seq ==="
  # idempotency: skip if already extracted ( FORCE=1 to re-download )
  img_dir="$KITTI360_DATA_ROOT/data_2d_raw/$seq/image_00/data_rect"
  velo_dir="$KITTI360_DATA_ROOT/data_3d_raw/$seq/velodyne_points/data"
  if [[ "${FORCE:-0}" != "1" && -d "$img_dir" && -n "$(ls -A "$img_dir" 2>/dev/null)" \
        && -d "$velo_dir" && -n "$(ls -A "$velo_dir" 2>/dev/null)" ]]; then
    log "already extracted, skipping download ( FORCE=1 to redo )"
    continue
  fi
  # rectified stereo pair (colour) + velodyne scans
  fetch_unzip "data_2d_raw/${seq}_image_00.zip" "${seq}_image_00.zip" "$KITTI360_DATA_ROOT/data_2d_raw"
  fetch_unzip "data_2d_raw/${seq}_image_01.zip" "${seq}_image_01.zip" "$KITTI360_DATA_ROOT/data_2d_raw"
  fetch_unzip "data_3d_raw/${seq}_velodyne.zip" "${seq}_velodyne.zip" "$KITTI360_DATA_ROOT/data_3d_raw"
  # per-sequence timestamps
  unzip -o -q "$TS_PERSP" "${seq}/*" -d "$KITTI360_DATA_ROOT/data_2d_raw"
  unzip -o -q "$TS_VELO"  "${seq}/*" -d "$KITTI360_DATA_ROOT/data_3d_raw"
done

if [[ "${KEEP_ZIPS:-0}" != "1" ]]; then
  log "removing downloaded zips ( KEEP_ZIPS=1 to keep )"
  rm -rf "$ZIP_DIR"
fi

log "layout ready under $KITTI360_DATA_ROOT"

# ------------------------------------------------------------------------ conversion
if [[ "${SKIP_CONVERT:-0}" == "1" ]]; then
  log "SKIP_CONVERT=1 -> stopping before py123d-conversion"
  exit 0
fi

command -v py123d-conversion >/dev/null || die "py123d-conversion not on PATH (pip install py123d)"

# Build hydra list overrides:  [a, b, c]
to_list() { local IFS=,; local arr=(); for x in $1; do arr+=("$(seq_name "$x")"); done; echo "[${arr[*]}]"; }
TRAIN_LIST="$(to_list "$TRAIN_SEQ")"
VAL_LIST="$(to_list "${VAL_SEQ:-}")"

log "converting -> $PY123D_DATA_ROOT/logs/kitti360_{train,val}"
log "  train_sequences=$TRAIN_LIST"
log "  val_sequences=$VAL_LIST"
py123d-conversion dataset=kitti360 \
  "dataset.parser.train_sequences=$TRAIN_LIST" \
  "dataset.parser.val_sequences=$VAL_LIST"

# --------------------------------------------------------------------------- verify
log "verifying converted split(s)"
python3 - <<'PY'
import os, numpy as np
from data import Py123dDataset
for split in ("kitti360_train", "kitti360_val"):
    try:
        ds = Py123dDataset(split_names=[split])
    except Exception as e:
        print(f"  {split}: (skipped: {e})"); continue
    n = len(ds)
    if n == 0:
        print(f"  {split}: 0 frames"); continue
    s = ds[min(400, n - 1)].to_stereo_sample(
        left_camera_id="pcam_stereo_l", right_camera_id="pcam_stereo_r")
    im = s.image_left.astype(np.int16)
    d = (abs(im[..., 0] - im[..., 1]).mean() + abs(im[..., 1] - im[..., 2]).mean()) / 2
    nb = 0 if s.boxes_3d is None else len(s.boxes_3d)
    print(f"  {split}: {n} frames | image {im.shape} "
          f"{'COLOR' if d > 0.5 else 'GRAYSCALE'} (chan-diff={d:.2f}) | "
          f"lidar {None if s.lidar_xyz is None else s.lidar_xyz.shape} | "
          f"boxes {nb} | baseline {float(s.calibration.stereo_baseline_m):.3f} m")
PY

log "done. Train with split_names=['kitti360_train'] (val: ['kitti360_val'])."
