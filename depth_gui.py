#!/usr/bin/env python3
"""
Interactive depth-based foreground / midground / background masking GUI.
- Fullscreen mode
- Auto-save masks on exit or Ctrl+C
- Deletes ~/Downloads/Figure_1.png at startup
"""

import argparse
import os
import sys
import signal
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.backend_bases import MouseButton


def safe_delete_download_figure():
    """Delete ~/Downloads/Figure_1.png if it exists."""
    try:
        fig_path = os.path.expanduser("~/Downloads/Figure_1.png")
        if os.path.exists(fig_path):
            os.remove(fig_path)
    except Exception:
        pass


def load_image(image_path):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def load_depth(depth_path, target_shape=None):
    ext = os.path.splitext(depth_path)[1].lower()
    if ext == ".npy":
        depth = np.load(depth_path)
    else:
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Could not read depth map: {depth_path}")
        if depth_raw.ndim == 3:
            depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
        depth = depth_raw.astype(np.float32)

    depth = np.asarray(depth, dtype=np.float32)

    if target_shape is not None and depth.shape != target_shape:
        depth = cv2.resize(depth, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

    d_min = np.min(depth)
    d_max = np.max(depth)
    if d_max > d_min:
        return (depth - d_min) / (d_max - d_min)
    return np.zeros_like(depth, dtype=np.float32)


def compute_masks(depth_norm, t1, t2, invert_depth=False):
    if t1 > t2:
        t1, t2 = t2, t1

    depth_used = depth_norm if invert_depth else 1.0 - depth_norm

    fg = depth_used <= t1
    mg = (depth_used > t1) & (depth_used <= t2)
    bg = depth_used > t2

    return fg, mg, bg


def make_overlay(image_rgb, fg_mask, mg_mask, bg_mask, alpha=0.5):
    color_fg = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    color_mg = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    color_bg = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    base = image_rgb.astype(np.float32) / 255.0
    overlay = base.copy()

    overlay[fg_mask] = (1 - alpha) * overlay[fg_mask] + alpha * color_fg
    overlay[mg_mask] = (1 - alpha) * overlay[mg_mask] + alpha * color_mg
    overlay[bg_mask] = (1 - alpha) * overlay[bg_mask] + alpha * color_bg

    return np.clip(overlay, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--preview-scale", type=float, default=0.25)
    args = parser.parse_args()
    safe_delete_download_figure()
    image_rgb = load_image(args.image)
    depth_norm = load_depth(args.depth, target_shape=image_rgb.shape[:2])
    scale = args.preview_scale
    H, W = image_rgb.shape[:2]
    prev_W, prev_H = int(W * scale), int(H * scale)

    preview_image = cv2.resize(image_rgb, (prev_W, prev_H), interpolation=cv2.INTER_AREA)
    preview_depth = cv2.resize(depth_norm, (prev_W, prev_H), interpolation=cv2.INTER_AREA)

    t1_init, t2_init = 0.3, 0.7

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    manager = plt.get_current_fig_manager()
    try:
        manager.full_screen_toggle()   # FULLSCREEN
    except Exception:
        pass

    plt.subplots_adjust(left=0.1, bottom=0.30)

    ax_img, ax_depth, ax_seg = axes
    ax_img.set_title("Original (preview)")
    ax_img.axis("off")
    ax_img.imshow(preview_image)

    ax_depth.set_title("Depth (preview)")
    ax_depth.axis("off")
    depth_im = ax_depth.imshow(preview_depth, cmap="magma")
    fig.colorbar(depth_im, ax=ax_depth, fraction=0.046, pad=0.04)

    fg_p, mg_p, bg_p = compute_masks(preview_depth, t1_init, t2_init)
    seg_im = ax_seg.imshow(make_overlay(preview_image, fg_p, mg_p, bg_p, alpha=args.alpha))
    ax_seg.axis("off")

    ax_t1 = plt.axes([0.1, 0.20, 0.8, 0.03])
    ax_t2 = plt.axes([0.1, 0.15, 0.8, 0.03])

    slider_t1 = Slider(ax_t1, "Foreground / Mid", 0.0, 1.0, valinit=t1_init)
    slider_t2 = Slider(ax_t2, "Mid / Background", 0.0, 1.0, valinit=t2_init)

    ax_check = plt.axes([0.01, 0.40, 0.12, 0.20])
    check = CheckButtons(ax_check,
                         ["Invert depth", "Log sliders"],
                         [False, False])

    state = {"invert": False, "log_sliders": False}

    # Helper function: convert linear->log and log->linear
    def lin2log(x):
        # Map [0,1] -> [1e-4, 1] in log scale
        return np.log10(1e-4 + x * (1 - 1e-4))

    def log2lin(y):
        # Inverse mapping
        return (10**y - 1e-4) / (1 - 1e-4)

    def rebuild_sliders():
        nonlocal slider_t1, slider_t2

        old_t1 = slider_t1.val
        old_t2 = slider_t2.val

        # Remove old sliders
        slider_t1.ax.remove()
        slider_t2.ax.remove()

        if state["log_sliders"]:
            # Use log-domain sliders: range = [-4, 0]
            t1_log = lin2log(old_t1)
            t2_log = lin2log(old_t2)

            ax_t1_new = plt.axes([0.1, 0.20, 0.8, 0.03])
            ax_t2_new = plt.axes([0.1, 0.15, 0.8, 0.03])

            slider_t1 = Slider(ax_t1_new, "Foreground / Mid (log)", -4, 0, valinit=t1_log)
            slider_t2 = Slider(ax_t2_new, "Mid / Background (log)", -4, 0, valinit=t2_log)

        else:
            # Linear sliders
            ax_t1_new = plt.axes([0.1, 0.20, 0.8, 0.03])
            ax_t2_new = plt.axes([0.1, 0.15, 0.8, 0.03])

            slider_t1 = Slider(ax_t1_new, "Foreground / Mid", 0.0, 1.0, valinit=old_t1)
            slider_t2 = Slider(ax_t2_new, "Mid / Background", 0.0, 1.0, valinit=old_t2)

        slider_t1.on_changed(update)
        slider_t2.on_changed(update)

    base_name = os.path.splitext(os.path.basename(args.image))[0]
    save_dir = "./masks"

    def get_slider_vals():
        if not state["log_sliders"]:
            return slider_t1.val, slider_t2.val
        else:
            return log2lin(slider_t1.val), log2lin(slider_t2.val)

    def save_full_res_masks():
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t1, t2 = get_slider_vals()
        invert = state["invert"]

        fg, mg, bg = compute_masks(depth_norm, t1, t2, invert_depth=invert)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_fg_mask.png"), fg.astype(np.uint8)*255)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_mg_mask.png"), mg.astype(np.uint8)*255)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_bg_mask.png"), bg.astype(np.uint8)*255)


    def update(_):
        t1, t2 = get_slider_vals()
        invert = state["invert"]

        fg_p, mg_p, bg_p = compute_masks(preview_depth, t1, t2, invert_depth=invert)
        seg_im.set_data(make_overlay(preview_image, fg_p, mg_p, bg_p, alpha=args.alpha))
        fig.canvas.draw_idle()

    slider_t1.on_changed(update)
    slider_t2.on_changed(update)

    def toggle_check(label):
        if label == "Invert depth":
            state["invert"] = not state["invert"]
            update(None)

        elif label == "Log sliders":
            state["log_sliders"] = not state["log_sliders"]
            rebuild_sliders()
            update(None)

    check.on_clicked(toggle_check)

    def handle_exit(*_):
        # save_full_res_masks()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    fig.canvas.mpl_connect("close_event", lambda evt: save_full_res_masks())

    # Keyboard save
    fig.canvas.mpl_connect("key_press_event",
                           lambda event: save_full_res_masks() if event.key == "s" else None)

    print("Controls:")
    print("  - Adjust sliders (linear or log mode)")
    print("  - Closing the window or Cmd+C auto-saves")

    plt.show()


if __name__ == "__main__":
    main()
