# How to start the Viser viewer

`py123d-viser` is py123d's 3D web GUI (lidar + boxes + map + camera frustums).
Already installed (`viser 1.0.24`). It reads dataset paths from env vars.

## Launch (data already in `data/`)

```bash
cd /home/leonardo/Desktop/AIRO/stereo-lidar-perception
export PY123D_DATA_ROOT="$PWD/data"   # converted logs/maps
export AV2_DATA_ROOT="$PWD/data"      # raw sensor blobs ($AV2_DATA_ROOT/sensor)
py123d-viser scene_filter=av2-sensor
```

Then open **http://localhost:8080**.

**Both env vars are required.** AV2 conversion stores only *relative* sensor
paths: `PY123D_DATA_ROOT` finds the logs, `AV2_DATA_ROOT` finds the images/lidar.
Omit the second → `AssertionError: Dataset path for sensor loading not found`.

## Common options

```bash
py123d-viser scene_filter=av2-sensor \
  'scene_filter.split_names=[av2-sensor_val]' \   # one split
  scene_filter.max_num_scenes=3                    # cap scenes (default: all, shuffled)
```

Also: `viser_config.server.port=8081`, `scene_filter.shuffle=false`.

> No data yet? `py123d-conversion dataset=av2-sensor-stream ...` downloads +
> converts into `PY123D_DATA_ROOT` (sensors embedded, so `AV2_DATA_ROOT` isn't
> needed), then launch as above.

---

Not to be confused with `python tests/test_data.py` — the lightweight matplotlib
viewer for *this* project's `data.py` loader (single camera, scroll frames). The
Viser viewer is the heavier full-scene 3D browser.
