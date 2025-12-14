# CS 445 Final Project: Depth-Aware Neural Style Transfer

Extending CNN-based style transfer (Gatys et al., CVPR 2016) with depth awareness for 3D-consistent stylization.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Baseline Style Transfer

```python
from style_transfer import stylize_image

stylized = stylize_image(
    content_path='data/images/house.jpg',
    style_path='data/styles/style_1.jpg',
    output_size=512,
    num_steps=300,
    style_weight=1e4,
    content_weight=1,
)
```

### Depth-Aware Style Transfer

```python
from depth_aware_style_transfer import load_precomputed_masks, blend_stylized_layers
from style_transfer import stylize_image

# Load pre-computed masks
masks = load_precomputed_masks(
    mask_dir='data/masks',
    image_name='house',
    target_shape=(512, 512),
    feather_radius=15
)

# Run baseline style transfer
stylized = stylize_image(content_path, style_path, output_size=512)

# Blend with depth-aware weights
style_strengths = {'foreground': 0.3, 'midground': 0.7, 'background': 1.0}
result = blend_stylized_layers(content_np, {'foreground': stylized, 'midground': stylized, 'background': stylized}, masks, style_strengths)
```

## Files

| File | Description |
|------|-------------|
| `style_transfer.py` | Baseline Gatys et al. implementation |
| `depth_aware_style_transfer.py` | Depth-aware blending pipeline |
| `depth_gui.py` | Interactive mask threshold tuning GUI |
| `eval.py` | Evaluation metrics (SSIM, PSNR, LPIPS) |
| `demo.ipynb` | Full pipeline demo (Colab) |
| `results_viewer.ipynb` | Interactive results browser |

## Dataset

```
data/
├── images/     # 100 content images
├── styles/     # 5 style images
├── masks/      # Pre-computed fg/mg/bg masks
└── depth/      # MiDaS depth maps
```

## Results

Pre-computed results in `results/`:
- `baseline/` - 500 baseline stylizations
- `depth_aware/` - 500 depth-aware stylizations

Evaluation metrics in `eval.csv`.

## References

- Gatys et al., "Image Style Transfer Using Convolutional Neural Networks", CVPR 2016
- [MiDaS](https://github.com/isl-org/MiDaS) for depth estimation
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) for perceptual evaluation
