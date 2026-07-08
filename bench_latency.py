#!/usr/bin/env python3
"""Latency Benchmarking Script for BEV Perception Pipeline Models.

Benchmarks inference latency (mean, std, FPS) and component-level breakdown
(LiDAR branch, Camera branch, Fusion block, Detection Head, and CenterPoint Decoder)
across multiple checkpoints/runs.

Usage examples:
    # Compare default runs (103240 vs 001858) on dataset validation frames
    python bench_latency.py

    # Benchmark specific checkpoints with 200 frames
    python bench_latency.py --runs runs/pipeline_c_yolo26_igev_20260708_103240 runs/pipeline_c_yolo26_igev_20260708_001858 --num-frames 200

    # Save results to JSON report
    python bench_latency.py --output latency_comparison.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import globals as G
from data import Py123dDataset, stereo_cache_root
from evaluation import CenterPointDecoder
from network import (
    CameraOnlyDetector,
    LidarOnlyDetector,
    MonoBEVConfig,
    MonoOnlyDetector,
    PipelineA,
    PipelineB,
    PipelineC,
    StereoBEVConfig,
    lidar_points,
)

DEFAULT_RUNS = [
    "runs/pipeline_c_yolo26_igev_20260708_103240",
    "runs/pipeline_c_yolo26_igev_20260708_001858",
]


def resolve_ckpt_and_config(run_path: str | Path) -> tuple[Path, dict[str, Any], str]:
    """Resolve a run folder or checkpoint file to (ckpt_path, config_dict, label)."""
    p = Path(run_path)
    if p.is_file():
        ckpt_path = p
        config_path = p.parent.parent / "config.json"
        label = p.parent.parent.name
    else:
        ckpt_path = p / "weights" / "best.pt"
        config_path = p / "config.json"
        label = p.name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        print(f"Warning: config.json not found at {config_path}, using defaults.")

    return ckpt_path, config, label


def load_model_from_config(
    ckpt_path: Path, config: dict[str, Any], dataset_root: str | Path | None = None
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Instantiate and load the model based on its run config."""
    model_type = config.get("model", "pipeline_c")
    matcher = config.get("matcher", "igev")
    cache_root = stereo_cache_root(dataset_root, matcher=matcher) if dataset_root else None

    stereo_cfg = StereoBEVConfig(
        img_backbone=config.get("camera_backbone", "efficientnet"),
        yolo_freeze=config.get("freeze_backbone", False),
        yolo_levels=config.get("yolo_levels", "p3"),
        use_depth_context=config.get("use_depth_context", False),
    )
    use_gaussian_attn = config.get("use_gaussian_attn", False)

    if model_type == "pipeline_c":
        model = PipelineC(
            stereo_cache_root=cache_root,
            stereo_cfg=stereo_cfg,
            use_gaussian_attn=use_gaussian_attn,
        )
    elif model_type == "pipeline_b":
        model = PipelineB(stereo_cache_root=cache_root, stereo_cfg=stereo_cfg)
    elif model_type == "pipeline_a":
        model = PipelineA(stereo_cache_root=cache_root, stereo_cfg=stereo_cfg)
    elif model_type == "camera":
        model = CameraOnlyDetector(stereo_cache_root=cache_root, stereo_cfg=stereo_cfg)
    elif model_type == "lidar":
        model = LidarOnlyDetector()
    elif model_type == "mono":
        mono_cfg = MonoBEVConfig(
            img_backbone=config.get("camera_backbone", "efficientnet"),
            yolo_freeze=config.get("freeze_backbone", False),
            yolo_levels=config.get("yolo_levels", "p3"),
        )
        model = MonoOnlyDetector(mono_cfg=mono_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)

    meta = {
        "model_type": model_type,
        "use_depth_context": stereo_cfg.use_depth_context,
        "use_gaussian_attn": use_gaussian_attn,
        "camera_backbone": stereo_cfg.img_backbone,
    }
    return model, meta


def bench_pipeline_model(
    model: torch.nn.Module,
    frames: list[Any],
    device: torch.device,
    warmup: int = 20,
    num_frames: int = 100,
    preload: bool = True,
) -> dict[str, dict[str, float]]:
    """Benchmark a Pipeline model with detailed sub-component timing on GPU."""
    model.to(device).eval()
    decoder = CenterPointDecoder()
    sample_kwargs = {"load_images": False, "point_mask": False}

    eval_frames = frames[: warmup + num_frames]
    if len(eval_frames) < warmup + 1:
        raise ValueError(
            f"Not enough frames ({len(frames)}) for warmup ({warmup}) + benchmark."
        )

    if preload:
        # Pre-load all samples into memory so ZERO disk I/O occurs during timing
        samples = [f.to_stereo_sample(**sample_kwargs) for f in eval_frames]
    else:
        samples = [f.to_stereo_sample(**sample_kwargs) for f in eval_frames]

    # Warmup runs
    with torch.no_grad():
        for sample in samples[:warmup]:
            out = model(sample, device=device)
            decoder(out["heatmap"].cpu(), out["offset"].cpu())

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timing containers
    component_times = {
        "lidar_branch": [],
        "camera_branch": [],
        "fusion_block": [],
        "detection_head": [],
        "net_forward": [],
        "decoder": [],
        "total_e2e": [],
    }

    evt_start = torch.cuda.Event(enable_timing=True)
    evt_lidar = torch.cuda.Event(enable_timing=True)
    evt_cam = torch.cuda.Event(enable_timing=True)
    evt_fus = torch.cuda.Event(enable_timing=True)
    evt_head = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for sample in samples[warmup:]:
            pts = lidar_points(sample)

            if device.type == "cuda" and hasattr(model, "lidar_branch") and hasattr(model, "camera_branch"):
                evt_start.record()
                bev_lidar = model.lidar_branch(pts, device=device)
                evt_lidar.record()
                bev_camera = model.camera_branch(sample, device=device)
                evt_cam.record()
                fused = model.detector.fusion(bev_camera, bev_lidar)
                evt_fus.record()
                out = model.detector.head(fused)
                evt_head.record()

                torch.cuda.synchronize()
                t0_dec = time.perf_counter()
                decoder(out["heatmap"].cpu(), out["offset"].cpu())
                t1_dec = time.perf_counter()

                lidar_ms = evt_start.elapsed_time(evt_lidar)
                cam_ms = evt_lidar.elapsed_time(evt_cam)
                fus_ms = evt_cam.elapsed_time(evt_fus)
                head_ms = evt_fus.elapsed_time(evt_head)
                net_ms = evt_start.elapsed_time(evt_head)
                dec_ms = (t1_dec - t0_dec) * 1000.0

                component_times["lidar_branch"].append(lidar_ms)
                component_times["camera_branch"].append(cam_ms)
                component_times["fusion_block"].append(fus_ms)
                component_times["detection_head"].append(head_ms)
                component_times["net_forward"].append(net_ms)
                component_times["decoder"].append(dec_ms)
                component_times["total_e2e"].append(net_ms + dec_ms)
            else:
                t0_net = time.perf_counter()
                out = model(sample, device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1_net = time.perf_counter()

                t0_dec = time.perf_counter()
                decoder(out["heatmap"].cpu(), out["offset"].cpu())
                t1_dec = time.perf_counter()

                net_ms = (t1_net - t0_net) * 1000.0
                dec_ms = (t1_dec - t0_dec) * 1000.0

                component_times["net_forward"].append(net_ms)
                component_times["decoder"].append(dec_ms)
                component_times["total_e2e"].append(net_ms + dec_ms)

    # Compute summary stats
    results = {}
    for k, vals in component_times.items():
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        results[k] = {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
        }

    # Add FPS based on total end-to-end mean and p50
    if "total_e2e" in results:
        mean_tot = results["total_e2e"]["mean_ms"]
        p50_tot = results["total_e2e"]["p50_ms"]
        results["summary"] = {
            "fps": 1000.0 / mean_tot if mean_tot > 0 else 0.0,
            "fps_p50": 1000.0 / p50_tot if p50_tot > 0 else 0.0,
            "mean_total_ms": mean_tot,
            "p50_total_ms": p50_tot,
        }

    return results


def print_comparison_table(all_results: dict[str, dict[str, Any]]) -> None:
    """Print a clean ASCII comparison table showing Mean ± Std and Median (p50)."""
    labels = list(all_results.keys())
    print("\n" + "=" * 132)
    print("INFERENCE LATENCY BENCHMARK COMPARISON (GPU / PyTorch)")
    print("=" * 132)

    for label in labels:
        meta = all_results[label].get("meta", {})
        print(
            f"  • {label[-15:]} ({label}): "
            f"model={meta.get('model_type')} | "
            f"backbone={meta.get('camera_backbone')} | "
            f"use_depth_context={meta.get('use_depth_context')} | "
            f"use_gaussian_attn={meta.get('use_gaussian_attn')}"
        )
    print("-" * 132)

    components = [
        ("lidar_branch", "LiDAR Branch"),
        ("camera_branch", "Camera Branch (StereoBEV)"),
        ("fusion_block", "Fusion Block"),
        ("detection_head", "Detection Head"),
        ("net_forward", "Total Network Forward"),
        ("decoder", "CenterPoint Decoder (CPU)"),
        ("total_e2e", "Total End-to-End Latency"),
    ]

    header = f"{'Component':<26}" + "".join(f"{lbl[-15:]:>29}" for lbl in labels)
    if len(labels) == 2:
        header += f"{'Δ Mean (L2-L1)':>20}{'Δ Med (L2-L1)':>20}"
    print(header)
    print("-" * 132)

    for comp_key, comp_name in components:
        row = f"{comp_name:<26}"
        means = []
        stds = []
        p50s = []
        for label in labels:
            stats = all_results[label]["timing"].get(comp_key)
            if stats:
                means.append(stats["mean_ms"])
                stds.append(stats["std_ms"])
                p50s.append(stats["p50_ms"])
                cell = f"{stats['mean_ms']:>6.2f}±{stats['std_ms']:<4.1f} (med:{stats['p50_ms']:>5.2f})"
                row += f"{cell:>29}"
            else:
                means.append(None)
                stds.append(None)
                p50s.append(None)
                row += f"{'N/A':>29}"

        if len(labels) == 2 and means[0] is not None and means[1] is not None:
            diff_m = means[1] - means[0]
            pct_m = (diff_m / means[0]) * 100.0 if means[0] > 0 else 0.0
            row += f"{diff_m:>11.2f} ms ({pct_m:+.1f}%)"
            diff_p = p50s[1] - p50s[0]
            pct_p = (diff_p / p50s[0]) * 100.0 if p50s[0] > 0 else 0.0
            row += f"{diff_p:>11.2f} ms ({pct_p:+.1f}%)"
        print(row)

    print("-" * 132)
    fps_row = f"{'Throughput (Mean FPS)':<26}"
    for label in labels:
        fps = all_results[label]["timing"].get("summary", {}).get("fps", 0.0)
        fps_row += f"{fps:>24.2f} FPS     "
    print(fps_row)
    fps_med_row = f"{'Throughput (Median FPS)':<26}"
    for label in labels:
        fps_p50 = all_results[label]["timing"].get("summary", {}).get("fps_p50", 0.0)
        fps_med_row += f"{fps_p50:>24.2f} FPS     "
    print(fps_med_row)
    print("=" * 132 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Latency Benchmark for BEV Perception Pipelines"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
        help="Run folders or .pt checkpoint paths to benchmark.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of validation frames to benchmark (after warmup).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup frames prior to timing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON benchmark report.",
    )
    parser.add_argument(
        "--no-preload",
        action="store_false",
        dest="preload",
        help="Disable preloading samples into RAM before timing.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Benchmarking on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    print("Loading validation dataset frames...")
    ds = Py123dDataset(split_names=["kitti360_val"])
    frames = [f for s in range(ds.scene_count) for f in ds.frames_in_scene(s)]
    print(f"Loaded {len(frames)} validation frames from {ds.data_root}.")

    all_results = {}

    for run_path in args.runs:
        ckpt_path, config, label = resolve_ckpt_and_config(run_path)
        print(f"\nLoading model [{label}] from {ckpt_path} ...")
        model, meta = load_model_from_config(ckpt_path, config, dataset_root=ds.data_root)

        print(
            f"Running benchmark on {args.num_frames} frames (warmup={args.warmup}, preload={args.preload}) ..."
        )
        timing_results = bench_pipeline_model(
            model,
            frames=frames,
            device=device,
            warmup=args.warmup,
            num_frames=args.num_frames,
            preload=args.preload,
        )

        all_results[label] = {"meta": meta, "timing": timing_results}

    print_comparison_table(all_results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved JSON benchmark report to: {out_path.absolute()}")


if __name__ == "__main__":
    main()
