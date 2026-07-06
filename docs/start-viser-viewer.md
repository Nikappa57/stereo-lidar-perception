# How to start the Viser viewer

`py123d-viser` is py123d's 3D web GUI (lidar + boxes + map + camera frustums).
Already installed (`viser 1.0.24`). It reads dataset paths from env vars.

## Launch (data already in `data/`)

```bash
cd /home/leonardo/Desktop/AIRO/stereo-lidar-perception
export PY123D_DATA_ROOT="$PWD/data"          # converted logs/maps
export KITTI360_DATA_ROOT="$PWD/KITTI-360"   # raw images + LiDAR blobs
py123d-viser scene_filter=kitti360
```

Then open **http://localhost:8080**.

**Both env vars are required.** The py123d conversion stores only *relative*
sensor paths: `PY123D_DATA_ROOT` finds the converted Arrow logs,
`KITTI360_DATA_ROOT` finds the original images/LiDAR scans.
Omit the second → `AssertionError: Dataset path for sensor loading not found`.

## Common options

```bash
py123d-viser scene_filter=kitti360 \
  'scene_filter.split_names=[kitti360_val]' \   # one split
  scene_filter.max_num_scenes=3                  # cap scenes (default: all, shuffled)
```

Also: `viser_config.server.port=8081`, `scene_filter.shuffle=false`.

> No data yet? Run `scripts/get_kitti360.sh` to download, extract and convert
> the KITTI-360 sequences into `data/logs/kitti360_{train,val}/`, then launch
> as above. See the main README for details.

---

Not to be confused with `python tests/test_data.py` — the lightweight matplotlib
viewer for *this* project's `data.py` loader (single camera, scroll frames). The
Viser viewer is the heavier full-scene 3D browser.
