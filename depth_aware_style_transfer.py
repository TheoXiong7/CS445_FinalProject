"""
CS 445 Final Project

Depth-Aware Style Transfer Pipeline
Integrates MiDaS depth estimation with Gatys et al. style transfer
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, Dict, Union
import os

# Import baseline style transfer
from style_transfer import (
    stylize_image,
    load_image,
    tensor_to_image,
    device
)


def load_midas_model(model_type: str = "DPT_Large") -> Tuple[torch.nn.Module, any]:
    """Load MiDaS depth estimation model.

    Args:
        model_type: One of "DPT_Large", "DPT_Hybrid", "MiDaS_small"

    Returns:
        model: MiDaS model
        transform: MiDaS transform for preprocessing
    """
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform


def estimate_depth(
    image: Union[str, np.ndarray],
    midas_model: torch.nn.Module,
    midas_transform: any
) -> np.ndarray:
    """Estimate depth map for an image using MiDaS.

    Args:
        image: Path to image or numpy array [H, W, 3] in RGB
        midas_model: Loaded MiDaS model
        midas_transform: MiDaS preprocessing transform

    Returns:
        depth_map: Normalized depth map [H, W] in range [0, 1]
                   where 0 = far, 1 = near
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image

    original_shape = img.shape[:2]

    # Transform and predict
    input_batch = midas_transform(img).to(device)

    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=original_shape,
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # Normalize to [0, 1]
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max > depth_min:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth)

    return depth


def compute_depth_masks(
    depth_map: np.ndarray,
    fg_threshold: float = 0.6,
    bg_threshold: float = 0.3,
    feather_radius: int = 15
) -> Dict[str, np.ndarray]:
    """Compute soft masks for foreground, midground, and background.

    Args:
        depth_map: Normalized depth [H, W] where 1 = near, 0 = far
        fg_threshold: Depth above this is foreground
        bg_threshold: Depth below this is background
        feather_radius: Gaussian blur radius for soft mask edges

    Returns:
        Dictionary with 'foreground', 'midground', 'background' soft masks
    """
    # Hard masks
    fg_hard = (depth_map >= fg_threshold).astype(np.float32)
    bg_hard = (depth_map <= bg_threshold).astype(np.float32)
    mg_hard = ((depth_map > bg_threshold) & (depth_map < fg_threshold)).astype(np.float32)

    # Soft masks with feathering for smooth blending
    if feather_radius > 0:
        kernel_size = feather_radius * 2 + 1
        fg_soft = cv2.GaussianBlur(fg_hard, (kernel_size, kernel_size), 0)
        bg_soft = cv2.GaussianBlur(bg_hard, (kernel_size, kernel_size), 0)
        mg_soft = cv2.GaussianBlur(mg_hard, (kernel_size, kernel_size), 0)
    else:
        fg_soft, mg_soft, bg_soft = fg_hard, mg_hard, bg_hard

    # Normalize so masks sum to 1 at each pixel
    total = fg_soft + mg_soft + bg_soft
    total = np.maximum(total, 1e-8)

    return {
        'foreground': fg_soft / total,
        'midground': mg_soft / total,
        'background': bg_soft / total
    }


def blend_stylized_layers(
    original: np.ndarray,
    stylized_layers: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    style_strengths: Dict[str, float] = None
) -> np.ndarray:
    """Blend stylized layers using depth masks.

    Args:
        original: Original image [H, W, 3]
        stylized_layers: Dict with 'foreground', 'midground', 'background' stylized images
        masks: Soft masks for each layer
        style_strengths: Blending strength per layer (0 = original, 1 = fully stylized)

    Returns:
        Blended output image [H, W, 3]
    """
    if style_strengths is None:
        # Default: more stylization in background, less in foreground
        style_strengths = {
            'foreground': 0.3,
            'midground': 0.7,
            'background': 1.0
        }

    output = np.zeros_like(original, dtype=np.float32)

    for layer in ['foreground', 'midground', 'background']:
        mask = masks[layer][:, :, np.newaxis]  # [H, W, 1]
        stylized = stylized_layers[layer].astype(np.float32)
        strength = style_strengths[layer]

        # Blend between original and stylized based on strength
        blended = (1 - strength) * original.astype(np.float32) + strength * stylized

        # Apply mask
        output += mask * blended

    return np.clip(output, 0, 255).astype(np.uint8)


def depth_aware_style_transfer(
    content_path: str,
    style_path: str,
    output_size: int = 512,
    num_steps: int = 200,
    style_weight: float = 1e4,
    content_weight: float = 1,
    fg_threshold: float = 0.6,
    bg_threshold: float = 0.3,
    feather_radius: int = 15,
    style_strengths: Dict[str, float] = None,
    midas_model: torch.nn.Module = None,
    midas_transform: any = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Run depth-aware style transfer.

    Args:
        content_path: Path to content image
        style_path: Path to style image
        output_size: Maximum dimension of output
        num_steps: Style transfer optimization steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        fg_threshold: Depth threshold for foreground
        bg_threshold: Depth threshold for background
        feather_radius: Feathering for mask edges
        style_strengths: Dict with stylization strength per layer
        midas_model: Pre-loaded MiDaS model (loads if None)
        midas_transform: Pre-loaded MiDaS transform
        verbose: Print progress

    Returns:
        result: Final blended image [H, W, 3]
        depth_map: Estimated depth map [H, W]
        masks: Dictionary of soft masks
    """
    if style_strengths is None:
        style_strengths = {
            'foreground': 0.3,
            'midground': 0.7,
            'background': 1.0
        }

    # Load MiDaS if not provided
    if midas_model is None:
        if verbose:
            print("Loading MiDaS depth estimation model...")
        midas_model, midas_transform = load_midas_model("DPT_Large")

    # Load and resize content image
    content_img = Image.open(content_path).convert('RGB')
    ratio = output_size / max(content_img.size)
    new_size = tuple(int(dim * ratio) for dim in content_img.size)
    content_img = content_img.resize(new_size, Image.LANCZOS)
    content_np = np.array(content_img)

    if verbose:
        print(f"Content image size: {content_np.shape[:2]}")

    # Estimate depth
    if verbose:
        print("Estimating depth...")
    depth_map = estimate_depth(content_np, midas_model, midas_transform)

    # Compute masks
    if verbose:
        print("Computing depth masks...")
    masks = compute_depth_masks(
        depth_map,
        fg_threshold=fg_threshold,
        bg_threshold=bg_threshold,
        feather_radius=feather_radius
    )

    # Run style transfer once (we'll use the same stylized output,
    # but blend with different strengths per layer)
    if verbose:
        print(f"Running style transfer ({num_steps} steps)...")

    stylized = stylize_image(
        content_path=content_path,
        style_path=style_path,
        output_size=output_size,
        num_steps=num_steps,
        style_weight=style_weight,
        content_weight=content_weight,
        verbose=verbose
    )

    # Resize stylized to match content if needed
    if stylized.shape[:2] != content_np.shape[:2]:
        stylized = cv2.resize(stylized, (content_np.shape[1], content_np.shape[0]))

    # Use same stylized image for all layers (different blending strengths)
    stylized_layers = {
        'foreground': stylized,
        'midground': stylized,
        'background': stylized
    }

    # Blend layers
    if verbose:
        print("Blending layers...")
    result = blend_stylized_layers(
        content_np, stylized_layers, masks, style_strengths
    )

    if verbose:
        print("Done!")

    return result, depth_map, masks


def depth_aware_style_transfer_per_layer(
    content_path: str,
    style_path: str,
    output_size: int = 512,
    num_steps_per_layer: Dict[str, int] = None,
    style_weight_per_layer: Dict[str, float] = None,
    content_weight: float = 1,
    fg_threshold: float = 0.6,
    bg_threshold: float = 0.3,
    feather_radius: int = 15,
    midas_model: torch.nn.Module = None,
    midas_transform: any = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run style transfer with different parameters per depth layer.

    This version runs style transfer separately for each layer with different
    style weights, giving more control over per-layer stylization.

    Args:
        content_path: Path to content image
        style_path: Path to style image
        output_size: Maximum dimension of output
        num_steps_per_layer: Steps per layer, e.g., {'foreground': 100, ...}
        style_weight_per_layer: Style weight per layer
        content_weight: Weight for content loss
        fg_threshold: Depth threshold for foreground
        bg_threshold: Depth threshold for background
        feather_radius: Feathering for mask edges
        midas_model: Pre-loaded MiDaS model
        midas_transform: Pre-loaded MiDaS transform
        verbose: Print progress

    Returns:
        result: Final blended image
        depth_map: Estimated depth map
        masks: Dictionary of soft masks
        stylized_layers: Dictionary of per-layer stylized images
    """
    if num_steps_per_layer is None:
        num_steps_per_layer = {
            'foreground': 100,
            'midground': 200,
            'background': 300
        }

    if style_weight_per_layer is None:
        style_weight_per_layer = {
            'foreground': 1e3,   # Less stylization
            'midground': 5e3,   # Medium stylization
            'background': 1e4   # Full stylization
        }

    # Load MiDaS if not provided
    if midas_model is None:
        if verbose:
            print("Loading MiDaS depth estimation model...")
        midas_model, midas_transform = load_midas_model("DPT_Large")

    # Load and resize content image
    content_img = Image.open(content_path).convert('RGB')
    ratio = output_size / max(content_img.size)
    new_size = tuple(int(dim * ratio) for dim in content_img.size)
    content_img = content_img.resize(new_size, Image.LANCZOS)
    content_np = np.array(content_img)

    if verbose:
        print(f"Content image size: {content_np.shape[:2]}")

    # Estimate depth
    if verbose:
        print("Estimating depth...")
    depth_map = estimate_depth(content_np, midas_model, midas_transform)

    # Compute masks
    if verbose:
        print("Computing depth masks...")
    masks = compute_depth_masks(
        depth_map,
        fg_threshold=fg_threshold,
        bg_threshold=bg_threshold,
        feather_radius=feather_radius
    )

    # Run style transfer for each layer
    stylized_layers = {}
    for layer in ['foreground', 'midground', 'background']:
        if verbose:
            print(f"\nProcessing {layer} layer...")
            print(f"  Steps: {num_steps_per_layer[layer]}, Style weight: {style_weight_per_layer[layer]}")

        stylized = stylize_image(
            content_path=content_path,
            style_path=style_path,
            output_size=output_size,
            num_steps=num_steps_per_layer[layer],
            style_weight=style_weight_per_layer[layer],
            content_weight=content_weight,
            verbose=verbose
        )

        # Resize if needed
        if stylized.shape[:2] != content_np.shape[:2]:
            stylized = cv2.resize(stylized, (content_np.shape[1], content_np.shape[0]))

        stylized_layers[layer] = stylized

    # Blend layers (full stylization, different weights already applied during transfer)
    style_strengths = {'foreground': 1.0, 'midground': 1.0, 'background': 1.0}

    if verbose:
        print("\nBlending layers...")
    result = blend_stylized_layers(
        content_np, stylized_layers, masks, style_strengths
    )

    if verbose:
        print("Done!")

    return result, depth_map, masks, stylized_layers


def visualize_depth_layers(
    image: np.ndarray,
    depth_map: np.ndarray,
    masks: Dict[str, np.ndarray]
) -> np.ndarray:
    """Create a visualization of the depth layers.

    Args:
        image: Original image [H, W, 3]
        depth_map: Depth map [H, W]
        masks: Dictionary of soft masks

    Returns:
        Visualization image [H, W*4, 3]
    """
    H, W = image.shape[:2]

    # Colorize depth map
    depth_colored = cv2.applyColorMap(
        (depth_map * 255).astype(np.uint8),
        cv2.COLORMAP_MAGMA
    )
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    # Create masked views
    fg_view = image.copy()
    mg_view = image.copy()
    bg_view = image.copy()

    fg_mask = masks['foreground'][:, :, np.newaxis]
    mg_mask = masks['midground'][:, :, np.newaxis]
    bg_mask = masks['background'][:, :, np.newaxis]

    # Dim non-selected regions
    fg_view = (fg_view * fg_mask + (1 - fg_mask) * fg_view * 0.3).astype(np.uint8)
    mg_view = (mg_view * mg_mask + (1 - mg_mask) * mg_view * 0.3).astype(np.uint8)
    bg_view = (bg_view * bg_mask + (1 - bg_mask) * bg_view * 0.3).astype(np.uint8)

    # Add colored overlay
    fg_view = np.where(fg_mask > 0.5,
                       (0.7 * fg_view + 0.3 * np.array([255, 0, 0])).astype(np.uint8),
                       fg_view)
    mg_view = np.where(mg_mask > 0.5,
                       (0.7 * mg_view + 0.3 * np.array([0, 255, 0])).astype(np.uint8),
                       mg_view)
    bg_view = np.where(bg_mask > 0.5,
                       (0.7 * bg_view + 0.3 * np.array([0, 0, 255])).astype(np.uint8),
                       bg_view)

    # Stack horizontally
    vis = np.concatenate([depth_colored, fg_view, mg_view, bg_view], axis=1)

    return vis


def load_precomputed_masks(
    mask_dir: str,
    image_name: str,
    target_shape: Tuple[int, int],
    feather_radius: int = 15
) -> Dict[str, np.ndarray]:
    """Load pre-computed masks from disk.

    Args:
        mask_dir: Directory containing mask files
        image_name: Base name of the image (without extension)
        target_shape: (H, W) to resize masks to
        feather_radius: Gaussian blur radius for soft edges

    Returns:
        Dictionary with 'foreground', 'midground', 'background' soft masks
    """
    masks = {}
    for layer, suffix in [('foreground', 'fg'), ('midground', 'mg'), ('background', 'bg')]:
        mask_path = os.path.join(mask_dir, f"{image_name}_{suffix}_mask.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = mask.astype(np.float32) / 255.0
        else:
            # Default to zeros if mask not found
            mask = np.zeros(target_shape, dtype=np.float32)

        # Apply feathering
        if feather_radius > 0:
            kernel_size = feather_radius * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

        masks[layer] = mask

    # Normalize so masks sum to 1
    total = masks['foreground'] + masks['midground'] + masks['background']
    total = np.maximum(total, 1e-8)
    for layer in masks:
        masks[layer] = masks[layer] / total

    return masks


def depth_aware_style_transfer_with_masks(
    content_path: str,
    style_path: str,
    masks: Dict[str, np.ndarray],
    output_size: int = 512,
    num_steps: int = 200,
    style_weight: float = 1e4,
    content_weight: float = 1,
    style_strengths: Dict[str, float] = None,
    verbose: bool = True
) -> np.ndarray:
    """Run depth-aware style transfer with pre-computed masks.

    Args:
        content_path: Path to content image
        style_path: Path to style image
        masks: Pre-computed soft masks dict
        output_size: Maximum dimension of output
        num_steps: Style transfer optimization steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        style_strengths: Dict with stylization strength per layer
        verbose: Print progress

    Returns:
        result: Final blended image [H, W, 3]
    """
    if style_strengths is None:
        style_strengths = {
            'foreground': 0.3,
            'midground': 0.7,
            'background': 1.0
        }

    # Load and resize content image
    content_img = Image.open(content_path).convert('RGB')
    ratio = output_size / max(content_img.size)
    new_size = tuple(int(dim * ratio) for dim in content_img.size)
    content_img = content_img.resize(new_size, Image.LANCZOS)
    content_np = np.array(content_img)

    # Run style transfer
    if verbose:
        print(f"Running style transfer ({num_steps} steps)...")

    stylized = stylize_image(
        content_path=content_path,
        style_path=style_path,
        output_size=output_size,
        num_steps=num_steps,
        style_weight=style_weight,
        content_weight=content_weight,
        verbose=verbose
    )

    # Resize stylized to match content if needed
    if stylized.shape[:2] != content_np.shape[:2]:
        stylized = cv2.resize(stylized, (content_np.shape[1], content_np.shape[0]))

    # Use same stylized image for all layers
    stylized_layers = {
        'foreground': stylized,
        'midground': stylized,
        'background': stylized
    }

    # Blend layers
    if verbose:
        print("Blending layers...")
    result = blend_stylized_layers(
        content_np, stylized_layers, masks, style_strengths
    )

    return result


def process_single_pair(args):
    """Process a single content-style pair. Used for parallel processing."""
    content_path, style_path, masks, output_size, num_steps, style_weight, content_weight, style_strengths, output_path = args

    try:
        result = depth_aware_style_transfer_with_masks(
            content_path=content_path,
            style_path=style_path,
            masks=masks,
            output_size=output_size,
            num_steps=num_steps,
            style_weight=style_weight,
            content_weight=content_weight,
            style_strengths=style_strengths,
            verbose=False
        )
        Image.fromarray(result).save(output_path)
        return output_path, True, None
    except Exception as e:
        return output_path, False, str(e)


# Demo / test
if __name__ == '__main__':
    dataset_path = 'style-transfer-dataset'
    content_path = os.path.join(dataset_path, 'contents', 'content_38.jpg')
    style_path = os.path.join(dataset_path, 'styles', 'style_18.jpg')

    if os.path.exists(content_path) and os.path.exists(style_path):
        print("Running depth-aware style transfer demo...")

        result, depth_map, masks = depth_aware_style_transfer(
            content_path=content_path,
            style_path=style_path,
            output_size=512,
            num_steps=200,
            style_weight=1e4,
            content_weight=1,
            fg_threshold=0.6,
            bg_threshold=0.3,
            style_strengths={
                'foreground': 0.3,
                'midground': 0.7,
                'background': 1.0
            },
            verbose=True
        )

        # Save result
        output_path = 'depth_aware_stylized.jpg'
        Image.fromarray(result).save(output_path)
        print(f"Saved depth-aware stylized image to {output_path}")

        # Save depth visualization
        content_img = np.array(Image.open(content_path).convert('RGB'))
        content_img = cv2.resize(content_img, (result.shape[1], result.shape[0]))
        vis = visualize_depth_layers(content_img, depth_map, masks)
        vis_path = 'depth_layers_vis.jpg'
        Image.fromarray(vis).save(vis_path)
        print(f"Saved depth layers visualization to {vis_path}")
    else:
        print(f"Dataset not found at {dataset_path}")
