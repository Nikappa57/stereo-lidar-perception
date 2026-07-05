# Visualize preprocessing results + network debugging helpers

from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt

from data import StereoSample, frustum_points, voxel_grid, cluster_points
from network import PillarConfig

#: Fixed per-class colours, shared by every figure (identity follows the
#: entity — same order as globals.CLASSES everywhere).
CLASS_COLORS = np.array([[0.15, 0.6, 1.0], [1.0, 0.55, 0.1], [0.2, 1.0, 0.3]])


# --------------------------------------------------------------------------- #
# Network debugging — capture and plot intermediate outputs
# --------------------------------------------------------------------------- #
@contextmanager
def record_activations(model, layer_names):
    """Capture submodule outputs by dotted path during forward passes.

    Registers forward hooks on each named submodule (``model.get_submodule``
    path, e.g. ``"camera_branch.model.context_head"``) and stores a detached
    CPU copy of its output under that name; hooks are removed on exit, so the
    model is untouched afterwards. Works on any ``nn.Module``. Usage::

        pipe = PipelineA().eval()
        with record_activations(pipe, [
                "camera_branch.model.backbone",      # (1, 256, H/8, W/8)
                "camera_branch.model.context_head",  # (1, 64, H/8, W/8)
                "lidar_branch.pfn",                  # (P, 64) pillar features
                "detector.fusion",                   # (1, 128, nx, ny)
        ]) as acts:
            with torch.no_grad():
                pipe(sample, device=device)
        acts["detector.fusion"].shape

    Print ``dict(model.named_modules()).keys()`` to discover valid paths.
    """
    acts, handles = {}, []

    def _make_hook(name):
        def _hook(module, args, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(out, dict):  # e.g. CenterPointHead returns a dict
                acts[name] = {k: v.detach().cpu() for k, v in out.items()}
            else:
                acts[name] = out.detach().cpu()
        return _hook

    for name in layer_names:
        module = model.get_submodule(name)
        handles.append(module.register_forward_hook(_make_hook(name)))
    try:
        yield acts
    finally:
        for h in handles:
            h.remove()


def visualize_pipeline_debug(pipeline, sample, device=None,
                             save_path: str | None = None):
    """One debug forward of a :class:`network.Pipeline` → panel of every stage.

    Panels (BEV ones on the shared ego grid, GT centres overlaid): left RGB |
    camera BEV | LiDAR BEV | fused BEV (channel-collapsed to |mean|) | head
    heatmap (sigmoid, max class) | offset magnitude. Untrained weights give
    structured-but-meaningless maps — the point is checking data flow, grid
    alignment and where activations live, not detection quality.

    Pass ``save_path`` to write a PNG instead of showing interactively.
    """
    import torch

    import globals as G

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.debug = True
    pipeline.to(device).eval()
    with torch.no_grad():
        pipeline(sample, device=device)
    inter = pipeline.intermediates

    (x_min, x_max), (y_min, y_max) = G.X_RANGE, G.Y_RANGE
    extent = [y_min, y_max, x_min, x_max]

    # in-grid GT centres, for the alignment check
    centres = sample.boxes_3d_ego[:, :2] if len(sample.boxes_3d_ego) else np.zeros((0, 2))
    m = ((centres[:, 0] >= x_min) & (centres[:, 0] < x_max) &
         (centres[:, 1] >= y_min) & (centres[:, 1] < y_max))
    centres = centres[m]

    def _panel(ax, feat, title, cmap):
        lo, hi = float(np.percentile(feat, 1)), float(np.percentile(feat, 99))
        if hi <= lo:
            lo, hi = float(feat.min()), float(feat.max()) + 1e-9
        im = ax.imshow(feat, origin="lower", cmap=cmap, vmin=lo, vmax=hi,
                       extent=extent, aspect="auto")
        if len(centres):
            ax.scatter(centres[:, 1], centres[:, 0], s=40, facecolors="none",
                       edgecolors="cyan", lw=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Y lateral (m)")
        ax.set_ylabel("X forward (m)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig, ax = plt.subplots(2, 3, figsize=(17, 11))
    fig.suptitle(f"{type(pipeline).__name__} debug forward | "
                 f"{sample.dataset} iter={sample.iteration} | GT rings = "
                 f"ego-frame box centres", fontsize=13, fontweight="bold")

    ax[0, 0].imshow(sample.image_left)
    ax[0, 0].set_title("Left RGB", fontsize=10)
    ax[0, 0].axis("off")

    _panel(ax[0, 1], inter["bev_camera"].abs().mean(0).numpy(),
           f"Camera BEV ({inter['bev_camera'].shape[0]} ch, |mean|)", "plasma")
    _panel(ax[0, 2], inter["bev_lidar"].abs().mean(0).numpy(),
           f"LiDAR BEV ({inter['bev_lidar'].shape[0]} ch, |mean|)", "viridis")
    _panel(ax[1, 0], inter["fused"].abs().mean(0).numpy(),
           f"Fused BEV ({inter['fused'].shape[0]} ch, |mean|)", "magma")
    _panel(ax[1, 1], inter["heatmap"].sigmoid().amax(0).numpy(),
           "Head heatmap (max class, σ)", "inferno")
    _panel(ax[1, 2], inter["offset"].norm(dim=0).numpy(),
           "Offset magnitude", "cividis")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return save_path
    plt.show()



# Utils for preprocessing visualizations


def visualize_bev(sample: StereoSample, cmap: str = "inferno") -> None:
    """Show a top-down LiDAR density BEV with the GT box outlines on top.

    The grid (x/y range, resolution) is taken from :class:`PillarConfig` so the
    picture is aligned with the LiDAR/camera BEV branches. Boxes are drawn from
    ``boxes_3d_ego`` — the ego-frame copy that matches the grid; ``boxes_3d`` is
    in the global frame and would land in the wrong cells here.
    """
    cfg = PillarConfig()
    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    res = cfg.pillar_size
    nx, ny = cfg.grid_size

    # Point density per cell (ix along x/forward, iy along y/lateral).
    grid = np.zeros((nx, ny), dtype=np.float32)
    pts = sample.lidar_xyz
    if pts.shape[0]:
        m = (pts[:, 0] >= x_min) & (pts[:, 0] < x_max) & (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
        p = pts[m]
        ix = ((p[:, 0] - x_min) / res).astype(np.int64)
        iy = ((p[:, 1] - y_min) / res).astype(np.int64)
        np.add.at(grid, (ix, iy), 1.0)

    plt.figure(figsize=(6, 8))
    # extent lets us plot boxes directly in metres; origin="lower" -> x up, y right.
    plt.imshow(np.log1p(grid), origin="lower", cmap=cmap,
               extent=[y_min, y_max, x_min, x_max], aspect="auto")
    boxes = sample.boxes_3d_ego
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            cx, cy, l, w = b[0], b[1], b[7], b[8]
            qw, qx, qy, qz = b[3], b[4], b[5], b[6]
            heading = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2])
            dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2])
            xs = cx + dx * cos_h - dy * sin_h
            ys = cy + dx * sin_h + dy * cos_h
            plt.plot(ys, xs, "r-", lw=1)  # (lateral, forward) to match extent
    plt.xlabel("Y lateral (m)")
    plt.ylabel("X forward (m)")
    plt.title("LiDAR density BEV + GT boxes (ego frame)")
    plt.show()


def visualize_encoded_targets(sample: StereoSample, save_path: str | None = None):
    """End-to-end sanity panel for the BEV target encoder + decoder.

    Four panels, all in the ego BEV frame (Y lateral horizontal, X forward
    vertical) so they overlay:

    1. left RGB image (the scene),
    2. LiDAR density BEV with oriented GT boxes (where the objects really are),
    3. the **encoded** heatmap (max over classes) with GT centres,
    4. the **decoded** predictions vs GT centres over the LiDAR BEV.

    If the Gaussian blobs and the decoded ``x`` markers land on the LiDAR
    objects / GT rings, the encode→decode round-trip and the frames all agree.
    Pass ``save_path`` to write a PNG instead of showing interactively.
    """
    import torch

    import globals as G
    from train import TargetEncoder
    from evaluation import CenterPointDecoder

    cfg = PillarConfig()
    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    res = cfg.pillar_size
    nx, ny = cfg.grid_size
    extent = [y_min, y_max, x_min, x_max]
    colors = CLASS_COLORS

    def _cls_color(arr):
        return colors[np.clip(np.asarray(arr, int), 0, len(colors) - 1)]

    # --- GT boxes: remap labels, keep in-grid + kept classes --------------
    boxes = sample.boxes_3d_ego
    idx = [G.class_index(l) for l in sample.boxes_3d_labels]
    keep = np.array([(c is not None) and (x_min <= b[0] < x_max) and
                     (y_min <= b[1] < y_max) for b, c in zip(boxes, idx)])
    centres = np.array([[b[0], b[1]] for b, k in zip(boxes, keep) if k]).reshape(-1, 2)

    # --- encode + decode --------------------------------------------------
    bx = torch.as_tensor(boxes, dtype=torch.float32)[torch.from_numpy(keep)]
    lab = torch.tensor([c for c, k in zip(idx, keep) if k], dtype=torch.long)
    heatmap, offset = TargetEncoder().encode(bx, lab)
    pred = CenterPointDecoder(score_threshold=0.5)(
        heatmap.unsqueeze(0), offset.unsqueeze(0))[0]
    dec = pred["boxes_2d"].numpy().reshape(-1, 2)      # (M, 2) [x, y]
    dec_cls = pred["classes"].numpy()

    # --- LiDAR density ----------------------------------------------------
    grid = np.zeros((nx, ny), np.float32)
    pts = sample.lidar_xyz
    m = ((pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
         (pts[:, 1] >= y_min) & (pts[:, 1] < y_max))
    p = pts[m]
    np.add.at(grid, (((p[:, 0] - x_min) / res).astype(int),
                     ((p[:, 1] - y_min) / res).astype(int)), 1.0)

    fig, ax = plt.subplots(2, 2, figsize=(15, 13))

    # 1. RGB
    ax[0, 0].imshow(sample.image_left)
    ax[0, 0].set_title("Left RGB")
    ax[0, 0].axis("off")

    # 2. LiDAR + oriented GT boxes
    ax[0, 1].imshow(np.log1p(grid), origin="lower", cmap="bone", extent=extent,
                    aspect="auto")
    for b, c in zip(boxes, idx):
        if c is None or not (x_min <= b[0] < x_max and y_min <= b[1] < y_max):
            continue
        cx, cy, l, w = b[0], b[1], b[7], b[8]
        qw, qx, qy, qz = b[3], b[4], b[5], b[6]
        h = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        ch, sh = np.cos(h), np.sin(h)
        dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2])
        dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2])
        xs, ys = cx + dx * ch - dy * sh, cy + dx * sh + dy * ch
        ax[0, 1].plot(ys, xs, "-", color=colors[c], lw=1.3)
    ax[0, 1].set_title("LiDAR density BEV + GT boxes")
    ax[0, 1].set_xlabel("Y lateral (m)")
    ax[0, 1].set_ylabel("X forward (m)")

    # 3. encoded heatmap + oriented GT 3D boxes (to compare blob vs footprint)
    ax[1, 0].imshow(heatmap.amax(0).numpy(), origin="lower", cmap="hot",
                    extent=extent, aspect="auto")
    for b, c in zip(boxes, idx):
        if c is None or not (x_min <= b[0] < x_max and y_min <= b[1] < y_max):
            continue
        cx, cy, l, w = b[0], b[1], b[7], b[8]
        qw, qx, qy, qz = b[3], b[4], b[5], b[6]
        h = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        ch, sh = np.cos(h), np.sin(h)
        dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2])
        dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2])
        xs, ys = cx + dx * ch - dy * sh, cy + dx * sh + dy * ch
        ax[1, 0].plot(ys, xs, "-", color="cyan", lw=1.0)
    ax[1, 0].set_title("Encoded heatmap (max over classes) + GT 3D boxes")
    ax[1, 0].set_xlabel("Y lateral (m)")
    ax[1, 0].set_ylabel("X forward (m)")

    # 4. decoded vs GT over LiDAR
    ax[1, 1].imshow(np.log1p(grid), origin="lower", cmap="bone", extent=extent,
                    aspect="auto")
    if len(centres):
        ax[1, 1].scatter(centres[:, 1], centres[:, 0], s=95, facecolors="none",
                         edgecolors="cyan", lw=1.5, label="GT centre")
    if len(dec):
        ax[1, 1].scatter(dec[:, 1], dec[:, 0], s=28, c=_cls_color(dec_cls),
                         marker="x", lw=1.5, label="decoded")
    ax[1, 1].legend(loc="upper right")
    ax[1, 1].set_title(f"Decoded {len(dec)} vs {len(centres)} GT centres")
    ax[1, 1].set_xlabel("Y lateral (m)")
    ax[1, 1].set_ylabel("X forward (m)")

    handles = [plt.Line2D([0], [0], marker="s", color="w", markersize=11,
                          markerfacecolor=colors[i], label=name)
               for i, name in enumerate(G.CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=len(G.CLASSES),
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=90, bbox_inches="tight")
        plt.close(fig)
        return save_path
    plt.show()

def visualize_detections(sample: StereoSample, detections: dict,
                         save_path: str | None = None):
    """Decoded detections vs GT centres over the LiDAR density BEV.

    ``detections`` is one element of :class:`evaluation.CenterPointDecoder`
    output (``boxes_2d`` / ``scores`` / ``classes``). GT rings are the in-grid,
    class-remapped centres (same filter as training targets). Marker colour =
    class, following the :data:`globals.CLASSES` order.
    """
    import torch

    import globals as G
    from evaluation import frame_ground_truth

    (x_min, x_max), (y_min, y_max) = G.X_RANGE, G.Y_RANGE
    extent = [y_min, y_max, x_min, x_max]
    nx, ny = G.GRID_SIZE
    colors = CLASS_COLORS

    grid = np.zeros((nx, ny), np.float32)
    pts = sample.lidar_xyz
    m = ((pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
         (pts[:, 1] >= y_min) & (pts[:, 1] < y_max))
    p = pts[m]
    np.add.at(grid, (((p[:, 0] - x_min) / G.BEV_RES_M).astype(int),
                     ((p[:, 1] - y_min) / G.BEV_RES_M).astype(int)), 1.0)

    gt_xy, gt_cls = frame_ground_truth(sample)
    det_xy = detections["boxes_2d"]
    det_xy = det_xy.numpy() if isinstance(det_xy, torch.Tensor) else np.asarray(det_xy)
    det_cls = np.asarray(detections["classes"]).astype(int)

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(np.log1p(grid), origin="lower", cmap="bone", extent=extent,
              aspect="auto")
    if len(gt_xy):
        ax.scatter(gt_xy[:, 1], gt_xy[:, 0], s=95, facecolors="none",
                   edgecolors="cyan", lw=1.5, label=f"GT ({len(gt_xy)})")
    if len(det_xy):
        c = colors[np.clip(det_cls, 0, len(colors) - 1)]
        ax.scatter(det_xy[:, 1], det_xy[:, 0], s=30, c=c, marker="x", lw=1.5,
                   label=f"decoded ({len(det_xy)})")
    ax.legend(loc="upper right")
    ax.set_xlabel("Y lateral (m)")
    ax.set_ylabel("X forward (m)")
    ax.set_title(f"Detections vs GT | {sample.log_name[:8]}… "
                 f"iter={sample.iteration}")
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return save_path
    plt.show()


def _stereo_inputs_for(model, sample):
    """Fetch the exact ``(image, depth, K, T_cam2ego)`` the camera branch uses.

    Prefers the branch's precomputed cache (byte-identical to training); falls
    back to live SGBM from the sample's images.
    """
    from data import load_stereo_inputs, stereo_branch_inputs
    branch = getattr(model, "camera_branch", None) or getattr(model, "branch", None)
    cache_root = getattr(branch, "cache_root", None)
    if cache_root is not None:
        bundle = load_stereo_inputs(cache_root, sample.log_name, sample.iteration)
        if bundle is not None:
            return bundle
    return stereo_branch_inputs(sample)


def _backproject_depth(depth, K, T_cam2ego):
    """Every valid depth pixel → ego-frame xyz, using the splat's own geometry.

    Mirrors :func:`network._build_grounded_frustum` in numpy: ``xyz_ego =
    R·(K⁻¹·[u,v,1]·depth) + t``. Returns ``(P, 3)`` for the ``depth > 0`` pixels.
    """
    h, w = depth.shape
    us = np.arange(w) + 0.5
    vs = np.arange(h) + 0.5
    uu, vv = np.meshgrid(us, vs)
    uvh = np.stack([uu.ravel(), vv.ravel(), np.ones(h * w)], 0)  # (3, N)
    rays = np.linalg.inv(K.astype(np.float64)) @ uvh              # (3, N)
    d = depth.ravel().astype(np.float64)
    xyz_cam = rays * d                                            # (3, N)
    R, t = T_cam2ego[:3, :3], T_cam2ego[:3, 3:4]
    xyz_ego = (R.astype(np.float64) @ xyz_cam + t.astype(np.float64)).T  # (N, 3)
    return xyz_ego[d > 0]


def visualize_stereo_bev_diagnostic(model, sample, device=None,
                                    score_threshold: float = 0.1,
                                    save_path: str | None = None):
    """Split the camera failure into *seeing* vs *placing* — four stages.

    Answers "does the net see the object in the image but scatter it to the
    wrong BEV cell because the stereo depth is bad?". Panels:

    1. **Camera view** — the left image with 2D GT boxes (coloured by class).
       If a class is boxed here, it is in frame; the backbone gets to see it.
    2. **SGBM depth** — the exact depth map the splat consumes (0 = invalid,
       shown black). Thin/low-texture objects (pedestrians) drop out here.
    3. **BEV geometry** — the stereo depth back-projected to ego (grey) over the
       LiDAR cloud (blue) with GT centres (stars). This is the money panel: a
       star with LiDAR under it but no grey nearby = depth never reconstructed
       it; grey offset in range from the star = depth is biased. Placement, not
       perception, is the failure when the image panel *did* box the object.
    4. **Network BEV output** — the predicted heatmap (max over classes) with GT
       centres (stars) *and* the decoded detections (``x`` markers, coloured by
       class) that survive NMS + the ``score_threshold``. A bright heatmap blob
       with no ``x`` on it means the decoder's threshold, not the network,
       dropped it; lower ``score_threshold`` to confirm.

    The sample must be loaded with images (``load_images=True``).
    """
    import torch

    import globals as G
    from evaluation import CenterPointDecoder, frame_ground_truth

    device = device or torch.device("cpu")
    (x_min, x_max), (y_min, y_max) = G.X_RANGE, G.Y_RANGE
    extent = [y_min, y_max, x_min, x_max]

    image, depth, K, T = _stereo_inputs_for(model, sample)
    stereo_ego = _backproject_depth(depth, K, T)
    gt_xy, gt_cls = frame_ground_truth(sample)

    model.eval()
    with torch.no_grad():
        out = model(sample, device=device)
    heat = out["heatmap"].sigmoid().max(dim=1).values[0].cpu().numpy()  # (nx, ny)
    det = CenterPointDecoder(score_threshold=score_threshold)(
        out["heatmap"].cpu(), out["offset"].cpu())[0]
    det_xy = det["boxes_2d"].numpy()          # (D, 2) ego x, y
    det_cls = det["classes"].numpy().astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1) camera view + 2D GT boxes -----------------------------------------
    ax = axes[0, 0]
    ax.imshow(sample.image_left)
    labels = sample.boxes_3d_labels
    n_boxed = 0
    for box, bi in zip(sample.boxes_2d_left, sample.boxes_2d_left_box_indices):
        ci = G.class_index(labels[bi])
        if ci is None:
            continue
        x0, y0, x1, y1 = box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=CLASS_COLORS[ci], lw=1.8))
        n_boxed += 1
    ax.set_title(f"1 · camera view — {n_boxed} GT boxes in image")
    ax.axis("off")

    # 2) depth the splat consumes ------------------------------------------
    ax = axes[0, 1]
    dm = np.ma.masked_where(depth <= 0, depth)
    im = ax.imshow(dm, cmap="turbo")
    ax.set_facecolor("black")
    valid_pct = 100.0 * (depth > 0).mean()
    ax.set_title(f"2 · SGBM depth (splat input) — {valid_pct:.0f}% valid")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="metric depth (m)")

    # 3) BEV geometry: stereo cloud vs LiDAR vs GT -------------------------
    ax = axes[1, 0]
    lp = sample.lidar_xyz
    lm = ((lp[:, 0] >= x_min) & (lp[:, 0] < x_max) &
          (lp[:, 1] >= y_min) & (lp[:, 1] < y_max))
    ax.scatter(lp[lm, 1], lp[lm, 0], s=1, c="#3a7bd5", alpha=0.25,
               label="LiDAR")
    sm = ((stereo_ego[:, 0] >= x_min) & (stereo_ego[:, 0] < x_max) &
          (stereo_ego[:, 1] >= y_min) & (stereo_ego[:, 1] < y_max))
    ax.scatter(stereo_ego[sm, 1], stereo_ego[sm, 0], s=1, c="0.55", alpha=0.35,
               label="stereo depth")
    for ci in range(len(G.CLASSES)):
        m = gt_cls == ci
        if m.any():
            ax.scatter(gt_xy[m, 1], gt_xy[m, 0], s=140, marker="*",
                       c=[CLASS_COLORS[ci]], edgecolors="k", lw=0.6,
                       label=f"GT {G.CLASSES[ci]}")
    if len(det_xy):  # decoded detections over the geometry: do they land on grey?
        ax.scatter(det_xy[:, 1], det_xy[:, 0], s=55,
                   c=CLASS_COLORS[np.clip(det_cls, 0, len(CLASS_COLORS) - 1)],
                   marker="x", lw=1.8, label=f"decoded ({len(det_xy)})")
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel("Y lateral (m)")
    ax.set_ylabel("X forward (m)")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    ax.set_title("3 · BEV geometry — stereo/LiDAR/GT + decoded")

    # 4) network BEV output: heatmap + GT stars + decoded detections -------
    ax = axes[1, 1]
    ax.imshow(heat, origin="lower", cmap="inferno", extent=extent,
              aspect="auto", vmin=0, vmax=1)
    for ci in range(len(G.CLASSES)):
        m = gt_cls == ci
        if m.any():
            ax.scatter(gt_xy[m, 1], gt_xy[m, 0], s=120, marker="*",
                       facecolors="none", edgecolors=CLASS_COLORS[ci], lw=1.6,
                       label=f"GT {G.CLASSES[ci]}")
    if len(det_xy):
        ax.scatter(det_xy[:, 1], det_xy[:, 0], s=55,
                   c=CLASS_COLORS[np.clip(det_cls, 0, len(CLASS_COLORS) - 1)],
                   marker="x", lw=1.8, label=f"decoded ({len(det_xy)})")
    if len(gt_xy) or len(det_xy):
        ax.legend(loc="upper right", fontsize=8, markerscale=1.5)
    ax.set_xlabel("Y lateral (m)")
    ax.set_ylabel("X forward (m)")
    ax.set_title(f"4 · BEV output — heatmap + decoded (score≥{score_threshold:g})")

    fig.suptitle(f"stereo→BEV diagnostic | {sample.log_name[:8]}… "
                 f"iter={sample.iteration}", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return save_path
    plt.show()


def visualize_evaluation(report: dict, save_path: str | None = None):
    """Diagnostic panel for an :func:`evaluation.evaluate_model` report.

    Three views, all at the report's operating distance band (default 2 m):

    1. **PR curves** per class, with the F1-optimal operating point marked.
    2. **F1 vs confidence** per class — read the deployment threshold off the
       peak (dot = the score the report tables use).
    3. **Confusion matrix** (rows = GT, cols = predicted, "background" row =
       false positives, column = missed GT), counts annotated.

    Classes with no GT in the eval split are dropped from the curves but keep
    their FP column in the matrix. Pass ``save_path`` to write a PNG.
    """
    import globals as G

    op_thr = report["op_threshold_m"]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
    fig.suptitle(f"Detection diagnostics @{op_thr:g} m | "
                 f"{report['n_frames']} frames | mAP {report['mAP']:.3f}",
                 fontsize=13, fontweight="bold")

    # --- 1. PR curves ------------------------------------------------------
    for c, name in enumerate(G.CLASSES):
        row = report["per_class"][name]
        if not row["n_gt"]:
            continue
        curve, op = row["curves"][op_thr], row["ops"][op_thr]
        ax[0].plot(curve["recall"], curve["precision"], lw=2,
                   color=CLASS_COLORS[c], label=f"{name} (AP {row[op_thr]:.2f})")
        ax[0].plot(op["recall"], op["precision"], "o", ms=8,
                   color=CLASS_COLORS[c], mec="white", mew=1.5)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1.02)
    ax[0].set_xlabel("recall")
    ax[0].set_ylabel("precision")
    ax[0].set_title(f"Precision–recall @{op_thr:g} m (dot = F1-optimal)",
                    fontsize=10)
    ax[0].grid(alpha=0.25)
    ax[0].legend(loc="lower left", fontsize=9)

    # --- 2. F1 vs confidence ------------------------------------------------
    for c, name in enumerate(G.CLASSES):
        row = report["per_class"][name]
        if not row["n_gt"]:
            continue
        curve, op = row["curves"][op_thr], row["ops"][op_thr]
        p, r = curve["precision"], curve["recall"]
        f1 = 2 * p * r / np.maximum(p + r, 1e-9)
        ax[1].plot(curve["scores"], f1, lw=2, color=CLASS_COLORS[c], label=name)
        ax[1].plot(op["score"], op["f1"], "o", ms=8, color=CLASS_COLORS[c],
                   mec="white", mew=1.5)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("confidence threshold")
    ax[1].set_ylabel("F1")
    ax[1].set_title("F1 vs confidence (dot = chosen operating point)",
                    fontsize=10)
    ax[1].grid(alpha=0.25)
    ax[1].legend(loc="upper right", fontsize=9)

    # --- 3. confusion matrix -------------------------------------------------
    conf = report["confusion"]
    m, labels = conf["matrix"], conf["labels"]
    im = ax[2].imshow(m, cmap="Blues")
    ax[2].set_xticks(range(len(labels)), labels, rotation=30, ha="right",
                     fontsize=9)
    ax[2].set_yticks(range(len(labels)), labels, fontsize=9)
    ax[2].set_xlabel("predicted")
    ax[2].set_ylabel("ground truth")
    ax[2].set_title(f"Confusion @{op_thr:g} m (bg row = FP, bg col = missed)",
                    fontsize=10)
    thresh = m.max() / 2 if m.max() else 1
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax[2].text(j, i, str(m[i, j]), ha="center", va="center",
                       fontsize=9,
                       color="white" if m[i, j] > thresh else "black")
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return save_path
    plt.show()


def visualize_frustum(sample: StereoSample, box_index: int = 0) -> None:
    """Show frustum points colored by depth or intensity."""
    fpts = frustum_points(sample, box_index=box_index)
    if fpts.shape[0] == 0: 
        print("No points in frustum")
        return
    pts = fpts[:, :3]
    colors = np.ones_like(pts)
    if fpts.shape[1] >= 6:
        colors = fpts[:, 3:]  # RGB
    else:
        colors = (pts[:, 2:] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-6)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors, s=5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Frustum Points")
    plt.show()

def visualize_voxels(sample: StereoSample) -> None:
    """Show voxel grid as a sparse occupancy plot (requires more code for 3D scatter)."""
    vox = voxel_grid(sample)
    # Simple occupancy visualization (not full 3D scatter for brevity)
    plt.imshow(vox.features[0].max(axis=0), cmap="gray")
    plt.title("Voxel Occupancy (Max Intensity)")
    plt.show()

def visualize_clusters(sample: StereoSample) -> None:
    """Show points colored by cluster labels on a 2D projection."""
    lbls, stats = cluster_points(sample)
    pts = sample.lidar_xyz
    # Project to 2D BEV-like for visualization
    plt.scatter(pts[:, 0], pts[:, 1], c=lbls, s=10)
    plt.title("Clustered Points (2D Projection)")
    plt.show()

def visualize_images(sample: StereoSample) -> None:
    """Show the left and right stereo images side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if sample.image_left is not None:
        axes[0].imshow(sample.image_left)
    axes[0].set_title("Left Image")
    axes[0].axis('off')
    
    if sample.image_right is not None:
        axes[1].imshow(sample.image_right)
    axes[1].set_title("Right Image")
    axes[1].axis('off')
        
    plt.tight_layout()
    plt.show()

def visualize_pointcloud(sample: StereoSample) -> None:
    """Show the full 3D LiDAR point cloud."""
    pts = sample.lidar_xyz
    if pts.shape[0] == 0:
        print("No LiDAR points to show")
        return
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-6)
    step = max(1, pts.shape[0] // 30000)  # Subsample to keep rendering fast
    
    ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2], c=colors[::step], s=1, cmap='viridis')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"LiDAR Point Cloud (Ego Frame)")
    
    plt.show()
