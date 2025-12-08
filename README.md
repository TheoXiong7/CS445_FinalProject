# CS 445 Final Project: Depth-Aware Neural Style Transfer

Extending CNN-based style transfer (Gatys et al., CVPR 2016) with depth awareness for 3D-consistent stylization.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
# Baseline Model
from style_transfer import stylize_image

stylized = stylize_image(
    content_path='style-transfer-dataset/contents/content_1.jpg',
    style_path='style-transfer-dataset/styles/style_1.jpg',
    output_size=1920,
    num_steps=300,
    style_weight=1e4,
    content_weight=1,
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_size` | 512 | Maximum image dimension |
| `num_steps` | 300 | Optimization iterations |
| `style_weight` | 1e4 | Stylization strength (beta) |
| `content_weight` | 1 | Content preservation (alpha) |
| `content_layers` | `['conv4_2']` | VGG layers for content |
| `style_layers` | `['conv1_1', ..., 'conv5_1']` | VGG layers for style |
| `init_image` | `'content'` | Initialization: `'content'`, `'style'`, or `'random'` |

## Dataset

Images in `style-transfer-dataset/`:
- `contents/` - 50 content images
- `styles/` - 50 style images

## References

- Gatys et al., "Image Style Transfer Using Convolutional Neural Networks", CVPR 2016
