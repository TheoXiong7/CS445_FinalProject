"""
Evaluation utilities for depth-aware neural style transfer experiments.

The functions here focus on:
- Low-level similarity metrics (SSIM, PSNR, optional LPIPS)
- Style vs. content loss measurements using VGG19 features
- Depth-layer-aware diagnostics to quantify haloing at mask boundaries
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from style_transfer import VGG_MEAN, VGG_STD, get_vgg19_features, device

# Optional LPIPS dependency (https://github.com/richzhang/PerceptualSimilarity)
try:
    import lpips  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    lpips = None


ImageLike = Union[str, np.ndarray, torch.Tensor, Path]


def _resolve_device(device_override: Optional[Union[str, torch.device]]) -> torch.device:
    """Resolve a user-specified device or fall back to the module default."""
    if device_override is None:
        return device
    return torch.device(device_override)


def _to_tensor(
    image: ImageLike,
    target_shape: Optional[Tuple[int, int]] = None,
    max_size: Optional[int] = None,
) -> torch.Tensor:
    """Convert a path/ndarray/tensor to a normalized BCHW float tensor on the correct device."""
    return _to_tensor_with_device(image, target_shape, max_size, None)


def _to_tensor_with_device(
    image: ImageLike,
    target_shape: Optional[Tuple[int, int]] = None,
    max_size: Optional[int] = None,
    device_override: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Convert a path/ndarray/tensor to a normalized BCHW float tensor on the correct device."""
    target_device = _resolve_device(device_override)
    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
        pil_tensor = transforms.functional.pil_to_tensor(pil).float() / 255.0
    elif isinstance(image, Image.Image):
        pil_tensor = transforms.functional.pil_to_tensor(image).float() / 255.0
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.5:  # assume 0-255 input
            arr = arr / 255.0
        pil_tensor = torch.from_numpy(arr).permute(2, 0, 1)
    elif isinstance(image, torch.Tensor):
        pil_tensor = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if pil_tensor.dim() == 3 and pil_tensor.shape[0] != 3 and pil_tensor.shape[-1] == 3:
        # Convert HWC tensors to CHW
        pil_tensor = pil_tensor.permute(2, 0, 1)

    if pil_tensor.dim() == 3:
        pil_tensor = pil_tensor.unsqueeze(0)

    if pil_tensor.max() > 1.5:
        pil_tensor = pil_tensor / 255.0

    pil_tensor = pil_tensor.to(device=target_device, dtype=torch.float32)

    if max_size is not None:
        _, _, h, w = pil_tensor.shape
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            pil_tensor = F.interpolate(
                pil_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

    if target_shape is not None:
        pil_tensor = F.interpolate(
            pil_tensor,
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )

    return pil_tensor.clamp(0.0, 1.0)


def _to_numpy(
    image: ImageLike,
    target_shape: Optional[Tuple[int, int]] = None,
    max_size: Optional[int] = None,
    device_override: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """Convert inputs to uint8 numpy image in HWC format."""
    tensor = _to_tensor_with_device(
        image,
        target_shape=target_shape,
        max_size=max_size,
        device_override=device_override,
    )
    tensor = tensor.squeeze(0).detach().cpu()
    np_img = tensor.permute(1, 2, 0).numpy()
    np_img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
    return np_img


def _create_ssim_window(
    window_size: int,
    sigma: float,
    channel: int,
    device_override: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    target_device = _resolve_device(device_override)
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    window_2d = gauss[:, None] @ gauss[None, :]
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window.to(device=target_device)


def compute_ssim(
    prediction: ImageLike,
    target: ImageLike,
    window_size: int = 11,
    sigma: float = 1.5,
    max_size: Optional[int] = None,
    device_override: Optional[Union[str, torch.device]] = None,
) -> float:
    """Structural Similarity Index between two images."""
    pred = _to_tensor_with_device(
        prediction,
        max_size=max_size,
        device_override=device_override,
    )
    tgt = _to_tensor_with_device(
        target,
        target_shape=pred.shape[-2:],
        max_size=max_size,
        device_override=device_override,
    )

    channel = pred.size(1)
    window = _create_ssim_window(window_size, sigma, channel, device_override=device_override)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(tgt, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(tgt * tgt, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * tgt, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


def compute_psnr(
    prediction: ImageLike,
    target: ImageLike,
    max_size: Optional[int] = None,
    device_override: Optional[Union[str, torch.device]] = None,
) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    pred = _to_tensor_with_device(
        prediction,
        max_size=max_size,
        device_override=device_override,
    )
    tgt = _to_tensor_with_device(
        target,
        target_shape=pred.shape[-2:],
        max_size=max_size,
        device_override=device_override,
    )
    mse = F.mse_loss(pred, tgt)
    if mse.item() == 0:
        return float("inf")
    target_device = pred.device
    psnr = 20 * torch.log10(torch.tensor(1.0, device=target_device) / torch.sqrt(mse))
    return psnr.item()


def load_lpips_model(
    net: str = "alex", device_override: Optional[Union[str, torch.device]] = None
) -> Any:
    """Load LPIPS model if dependency is present."""
    if lpips is None:
        raise ImportError(
            "LPIPS is not installed. Install with `pip install lpips` to enable this metric."
        )
    target_device = _resolve_device(device_override)
    model = lpips.LPIPS(net=net).to(target_device)
    model.eval()
    return model


def compute_lpips(
    prediction: ImageLike,
    target: ImageLike,
    lpips_model: Optional[Any] = None,
    net: str = "alex",
    max_size: Optional[int] = None,
    device_override: Optional[Union[str, torch.device]] = None,
) -> float:
    """Compute LPIPS distance. Lower is closer."""
    if lpips_model is None:
        lpips_model = load_lpips_model(net=net, device_override=device_override)
    else:
        target_device = _resolve_device(device_override)
        try:
            if next(lpips_model.parameters()).device != target_device:
                lpips_model = lpips_model.to(target_device)
        except Exception:
            pass

    tgt_tensor = _to_tensor_with_device(
        target,
        max_size=max_size,
        device_override=device_override,
    )
    pred_tensor = _to_tensor_with_device(
        prediction,
        target_shape=tgt_tensor.shape[-2:],
        max_size=max_size,
        device_override=device_override,
    )

    # LPIPS expects [-1, 1]
    pred_tensor = pred_tensor * 2 - 1
    tgt_tensor = tgt_tensor * 2 - 1

    with torch.no_grad():
        dist = lpips_model(pred_tensor, tgt_tensor)
    return dist.mean().item()


def _normalize_for_vgg(x: torch.Tensor) -> torch.Tensor:
    return _normalize_for_vgg_with_device(x, None)


def _normalize_for_vgg_with_device(
    x: torch.Tensor, device_override: Optional[Union[str, torch.device]]
) -> torch.Tensor:
    target_device = _resolve_device(device_override)
    mean = VGG_MEAN.to(target_device)
    std = VGG_STD.to(target_device)
    return (x - mean) / std


def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    features = x.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


def _extract_features(
    x: torch.Tensor,
    layers: List[str],
    vgg: Optional[nn.Sequential] = None,
    device_override: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    target_device = _resolve_device(device_override)
    if vgg is None:
        vgg = get_vgg19_features()
    else:
        try:
            if next(vgg.parameters()).device != target_device:
                vgg = vgg.to(target_device)
        except StopIteration:
            pass

    layers = set(layers)
    feats: Dict[str, torch.Tensor] = {}

    conv_idx = 0
    block_idx = 1
    out = _normalize_for_vgg_with_device(x, device_override)

    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1
            name = f"conv{block_idx}_{conv_idx}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu{block_idx}_{conv_idx}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, (nn.AvgPool2d, nn.MaxPool2d)):
            name = f"pool{block_idx}"
            block_idx += 1
            conv_idx = 0
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn{block_idx}_{conv_idx}"
        else:
            name = f"layer{block_idx}_{conv_idx}"

        out = layer(out)
        if name in layers:
            feats[name] = out
        if len(feats) == len(layers):
            break

    return feats


def compute_style_content_losses(
    stylized: ImageLike,
    content: ImageLike,
    style: ImageLike,
    content_layers: Optional[List[str]] = None,
    style_layers: Optional[List[str]] = None,
    vgg: Optional[nn.Sequential] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    max_size: Optional[int] = None,
    device_override: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """Compute Gatys-style content/style losses and their ratio."""
    if content_layers is None:
        content_layers = ["conv4_2"]
    if style_layers is None:
        style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

    content_tensor = _to_tensor_with_device(
        content,
        max_size=max_size,
        device_override=device_override,
    )
    style_tensor = _to_tensor_with_device(
        style,
        target_shape=content_tensor.shape[-2:],
        max_size=max_size,
        device_override=device_override,
    )
    stylized_tensor = _to_tensor_with_device(
        stylized,
        target_shape=content_tensor.shape[-2:],
        max_size=max_size,
        device_override=device_override,
    )

    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).float()
        else:
            mask_tensor = mask.float()
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.to(content_tensor.device)
        mask_tensor = F.interpolate(
            mask_tensor,
            size=content_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # Keep masked regions but avoid zeroing everything else (fill with mean color)
        def apply_mask(img: torch.Tensor) -> torch.Tensor:
            mean_color = img.mean(dim=(2, 3), keepdim=True)
            return img * mask_tensor + (1 - mask_tensor) * mean_color
    else:
        apply_mask = lambda img: img

    with torch.no_grad():
        content_feats = _extract_features(
            apply_mask(content_tensor), content_layers, vgg, device_override=device_override
        )
        style_feats = _extract_features(
            apply_mask(style_tensor), style_layers, vgg, device_override=device_override
        )
        stylized_feats = _extract_features(
            apply_mask(stylized_tensor),
            list(set(content_layers + style_layers)),
            vgg,
            device_override=device_override,
        )

    content_loss = 0.0
    for layer in content_layers:
        content_loss += F.mse_loss(stylized_feats[layer], content_feats[layer])

    style_loss = 0.0
    for layer in style_layers:
        style_loss += F.mse_loss(_gram_matrix(stylized_feats[layer]), _gram_matrix(style_feats[layer]))

    ratio = style_loss.item() / (content_loss.item() + 1e-8)

    return {
        "content_loss": content_loss.item(),
        "style_loss": style_loss.item(),
        "style_to_content": ratio,
    }


def compute_boundary_artifact_score(
    stylized: ImageLike,
    content: ImageLike,
    mask: Union[np.ndarray, torch.Tensor],
    band_px: int = 3,
    max_size: Optional[int] = None,
    device_override: Optional[Union[str, torch.device]] = None,
) -> float:
    """Average absolute difference along mask boundaries (lower is better, fewer halos)."""
    stylized_np = _to_numpy(
        stylized,
        max_size=max_size,
        device_override=device_override,
    )
    content_np = _to_numpy(
        content,
        target_shape=stylized_np.shape[:2],
        max_size=max_size,
        device_override=device_override,
    )

    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    if mask_np.ndim == 3:
        mask_np = mask_np.squeeze()
    if mask_np.shape != stylized_np.shape[:2]:
        mask_np = cv2.resize(
            mask_np.astype(np.float32),
            (stylized_np.shape[1], stylized_np.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    mask_u8 = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)
    edges = cv2.Canny(mask_u8, 50, 150)
    if band_px > 1:
        kernel = np.ones((band_px, band_px), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    edge_idx = edges.astype(bool)
    if edge_idx.sum() == 0:
        return 0.0

    diff = np.abs(stylized_np.astype(np.float32) - content_np.astype(np.float32))
    return float(diff[edge_idx].mean())


def evaluate_stylization(
    stylized: ImageLike,
    content: ImageLike,
    style: ImageLike,
    depth_masks: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
    lpips_model: Optional[Any] = None,
    compute_lpips_metric: bool = False,
    boundary_band_px: int = 3,
    vgg: Optional[nn.Sequential] = None,
    max_size: Optional[int] = None,
    device_override: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Compute a collection of metrics for a single stylized result.

    Args:
        stylized: Predicted/stylized image.
        content: Content/reference image.
        style: Style image used for stylization.
        depth_masks: Optional dict of per-layer masks for depth-aware metrics.
        lpips_model: Pre-loaded LPIPS model (keeps GPU memory warm across calls).
        compute_lpips_metric: Whether to compute LPIPS (requires dependency).
        boundary_band_px: Width of boundary band (in pixels) for halo score.
        vgg: Optional pre-loaded VGG19 features module (reused across calls).
        max_size: Optional maximum long-side size (pixels) for evaluation to save memory.
        device_override: Force evaluation on a specific device (e.g., "cuda" or torch.device("cuda"));
            defaults to the style_transfer.device determined at import time.
    """
    target_device = _resolve_device(device_override)
    if vgg is None:
        vgg = get_vgg19_features()
    else:
        try:
            if next(vgg.parameters()).device != target_device:
                vgg = vgg.to(target_device)
        except StopIteration:
            pass

    results: Dict[str, Any] = {}
    with torch.no_grad():
        results["ssim"] = compute_ssim(
            stylized,
            content,
            max_size=max_size,
            device_override=target_device,
        )
        results["psnr"] = compute_psnr(
            stylized,
            content,
            max_size=max_size,
            device_override=target_device,
        )

        if compute_lpips_metric:
            try:
                results["lpips"] = compute_lpips(
                    stylized,
                    content,
                    lpips_model=lpips_model,
                    max_size=max_size,
                    device_override=target_device,
                )
            except ImportError:
                results["lpips"] = None
                results["lpips_note"] = "Install lpips to enable LPIPS metric."
        else:
            results["lpips"] = None

        results["style_content"] = compute_style_content_losses(
            stylized,
            content,
            style,
            vgg=vgg,
            max_size=max_size,
            device_override=target_device,
        )

        if depth_masks:
            per_layer: Dict[str, Any] = {}
            for name, mask in depth_masks.items():
                per_layer[name] = {
                    "style_content": compute_style_content_losses(
                        stylized,
                        content,
                        style,
                        mask=mask,
                        vgg=vgg,
                        max_size=max_size,
                        device_override=target_device,
                    ),
                    "boundary_artifact": compute_boundary_artifact_score(
                        stylized,
                        content,
                        mask,
                        band_px=boundary_band_px,
                        max_size=max_size,
                        device_override=target_device,
                    ),
                }
            results["depth_layers"] = per_layer

    return results


def evaluate_batch(
    predictions: List[ImageLike],
    contents: List[ImageLike],
    styles: List[ImageLike],
    depth_masks: Optional[List[Optional[Dict[str, Union[np.ndarray, torch.Tensor]]]]] = None,
    lpips_model: Optional[Any] = None,
    compute_lpips_metric: bool = False,
    vgg: Optional[nn.Sequential] = None,
    max_size: Optional[int] = None,
    device_override: Optional[Union[str, torch.device]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate a batch of stylizations.

    Pass `device_override="cuda"` (or a torch.device) to keep evaluation on GPU when available.
    Optionally pass a shared `vgg` or `lpips_model` on that device to amortize setup cost.
    Use `max_size` to downscale long edges before metrics to keep VRAM lower.
    """
    if vgg is None:
        vgg = get_vgg19_features()

    batch_results: List[Dict[str, Any]] = []
    mask_list = depth_masks if depth_masks is not None else [None] * len(predictions)
    for pred, cont, sty, masks in zip(predictions, contents, styles, mask_list):
        batch_results.append(
            evaluate_stylization(
                pred,
                cont,
                sty,
                depth_masks=masks,
                lpips_model=lpips_model,
                compute_lpips_metric=compute_lpips_metric,
                vgg=vgg,
                max_size=max_size,
                device_override=device_override,
            )
        )
    return batch_results


def demo_usage():
    """Small demo helper for the notebook."""
    print("Example usage inside demo.ipynb:")
    print(
        """
from eval import evaluate_stylization, load_lpips_model

metrics = evaluate_stylization(
    stylized_img,           # numpy/tensor/path
    content_img,
    style_img,
    depth_masks=masks,      # optional dict: {'foreground': mask, ...}
    max_size=768,           # optional: downscale long side for metrics to save VRAM
    compute_lpips_metric=False, # set True if lpips is installed
    device_override=device      # optional: force GPU/CPU for metrics
)
print(metrics)
"""
    )


if __name__ == "__main__":  # pragma: no cover
    warnings.filterwarnings("ignore")
    demo_usage()
