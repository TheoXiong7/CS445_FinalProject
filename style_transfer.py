"""
Baseline model
Theo Xiong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Optional, Tuple, List, Union
import copy


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# VGG19 normalization values (ImageNet)
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
VGG_STD = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)


class Normalization(nn.Module):
    """Normalize input images using ImageNet mean and std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ContentLoss(nn.Module):
    """Content loss module - computes MSE between feature maps."""

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    """Style loss module - computes MSE between Gram matrices."""

    def __init__(self, target_feature: torch.Tensor):
        super().__init__()
        self.target = self._gram_matrix(target_feature).detach()
        self.loss = 0

    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style representation.

        G_ij = sum_k F_ik * F_jk
        """
        b, c, h, w = x.size()
        features = x.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        # Normalize by number of elements
        return gram.div(b * c * h * w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gram = self._gram_matrix(x)
        self.loss = F.mse_loss(gram, self.target)
        return x


def get_vgg19_features() -> nn.Sequential:
    """Load pretrained VGG19 and return feature extraction layers only.

    Uses average pooling instead of max pooling as suggested in the paper.
    """
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

    # Replace max pooling with average pooling for smoother results
    for i, layer in enumerate(vgg.children()):
        if isinstance(layer, nn.MaxPool2d):
            vgg[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    # Freeze all parameters
    for param in vgg.parameters():
        param.requires_grad_(False)

    return vgg


def build_style_transfer_model(
    vgg: nn.Sequential,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    content_layers: List[str] = None,
    style_layers: List[str] = None
) -> Tuple[nn.Sequential, List[ContentLoss], List[StyleLoss]]:
    """Build the style transfer model with content and style loss modules.

    Args:
        vgg: Pretrained VGG19 feature extractor
        content_img: Content image tensor [1, 3, H, W]
        style_img: Style image tensor [1, 3, H, W]
        content_layers: Layers for content loss (default: ['conv4_2'])
        style_layers: Layers for style loss (default: ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])

    Returns:
        model: Sequential model with loss modules inserted
        content_losses: List of ContentLoss modules
        style_losses: List of StyleLoss modules
    """
    # Default layers from the paper
    if content_layers is None:
        content_layers = ['conv4_2']
    if style_layers is None:
        style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    # Normalization layer
    normalization = Normalization(VGG_MEAN, VGG_STD).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    # VGG19 layer naming convention
    conv_idx = 0
    block_idx = 1

    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1
            name = f'conv{block_idx}_{conv_idx}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu{block_idx}_{conv_idx}'
            layer = nn.ReLU(inplace=False)  # Replace in-place ReLU
        elif isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.MaxPool2d):
            name = f'pool{block_idx}'
            block_idx += 1
            conv_idx = 0
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn{block_idx}_{conv_idx}'
        else:
            name = f'layer{block_idx}_{conv_idx}'

        model.add_module(name, layer)

        # Add content loss
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{name}', content_loss)
            content_losses.append(content_loss)

        # Add style loss
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{name}', style_loss)
            style_losses.append(style_loss)

    # Trim layers after last content/style loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:i + 1]

    return model, content_losses, style_losses


def load_image(
    image_path: str,
    max_size: int = 512,
    shape: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """Load and preprocess an image for style transfer.

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (height or width)
        shape: Optional (H, W) to resize to specific dimensions

    Returns:
        Image tensor of shape [1, 3, H, W]
    """
    image = Image.open(image_path).convert('RGB')

    # Resize while maintaining aspect ratio
    if shape is None:
        size = max_size
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(size, Image.LANCZOS)
    else:
        image = image.resize((shape[1], shape[0]), Image.LANCZOS)

    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert output tensor to numpy image array.

    Args:
        tensor: Image tensor [1, 3, H, W] or [3, H, W]

    Returns:
        Numpy array [H, W, 3] with values in [0, 255]
    """
    image = tensor.cpu().clone().detach()
    if image.dim() == 4:
        image = image.squeeze(0)
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def run_style_transfer(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    num_steps: int = 300,
    style_weight: float = 1e4,
    content_weight: float = 1,
    content_layers: List[str] = None,
    style_layers: List[str] = None,
    init_image: str = 'content',
    verbose: bool = True
) -> torch.Tensor:
    """Run the style transfer optimization.

    Args:
        content_img: Content image tensor [1, 3, H, W]
        style_img: Style image tensor [1, 3, H, W]
        num_steps: Number of optimization steps
        style_weight: Weight for style loss (beta in paper)
        content_weight: Weight for content loss (alpha in paper)
        content_layers: Layers for content loss
        style_layers: Layers for style loss
        init_image: Initialization method ('content', 'style', or 'random')
        verbose: Print progress during optimization

    Returns:
        Stylized image tensor [1, 3, H, W]
    """
    # Load VGG19
    vgg = get_vgg19_features()

    # Build model with loss modules
    model, content_losses, style_losses = build_style_transfer_model(
        vgg, content_img, style_img, content_layers, style_layers
    )

    # Initialize output image
    if init_image == 'content':
        output_img = content_img.clone()
    elif init_image == 'style':
        output_img = style_img.clone()
    else:  # random
        output_img = torch.randn_like(content_img)

    output_img.requires_grad_(True)

    # Use L-BFGS optimizer as suggested in the paper
    optimizer = optim.LBFGS([output_img])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            # Clamp values to valid range
            with torch.no_grad():
                output_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(output_img)

            # Compute losses
            content_score = sum(cl.loss for cl in content_losses)
            style_score = sum(sl.loss for sl in style_losses)

            content_loss = content_weight * content_score
            style_loss = style_weight * style_score
            total_loss = content_loss + style_loss

            total_loss.backward()

            run[0] += 1
            if verbose and run[0] % 50 == 0:
                print(f'Step {run[0]}: Content Loss: {content_loss.item():.4f}, '
                      f'Style Loss: {style_loss.item():.4f}')

            return total_loss

        optimizer.step(closure)

    # Final clamp
    with torch.no_grad():
        output_img.clamp_(0, 1)

    return output_img


def stylize_image(
    content_path: Union[str, np.ndarray, torch.Tensor],
    style_path: Union[str, np.ndarray, torch.Tensor],
    output_size: int = 512,
    num_steps: int = 300,
    style_weight: float = 1e4,
    content_weight: float = 1,
    content_layers: List[str] = None,
    style_layers: List[str] = None,
    init_image: str = 'content',
    verbose: bool = True
) -> np.ndarray:
    """
    Args:
        content_path: Path to content image, numpy array [H,W,3], or tensor [1,3,H,W]
        style_path: Path to style image, numpy array [H,W,3], or tensor [1,3,H,W]
        output_size: Maximum dimension of output image
        num_steps: Number of optimization steps (more = better quality, slower)
        style_weight: Weight for style loss (higher = more stylization)
        content_weight: Weight for content loss (higher = preserve more content)
        content_layers: VGG layers for content (default: ['conv4_2'])
        style_layers: VGG layers for style (default: ['conv1_1'-'conv5_1'])
        init_image: 'content', 'style', or 'random'
        verbose: Print optimization progress

    Returns:
        Stylized image as numpy array [H, W, 3] with values in [0, 255]

    """
    # Load content image
    if isinstance(content_path, str):
        content_img = load_image(content_path, max_size=output_size)
    elif isinstance(content_path, np.ndarray):
        content_img = torch.from_numpy(content_path).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        content_img = content_img.to(device)
    else:
        content_img = content_path.to(device)

    # Get content image shape for style image
    _, _, h, w = content_img.shape

    # Load style image (resize to match content)
    if isinstance(style_path, str):
        style_img = load_image(style_path, shape=(h, w))
    elif isinstance(style_path, np.ndarray):
        style_img = torch.from_numpy(style_path).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        style_img = F.interpolate(style_img, size=(h, w), mode='bilinear', align_corners=False)
        style_img = style_img.to(device)
    else:
        style_img = F.interpolate(style_path, size=(h, w), mode='bilinear', align_corners=False)
        style_img = style_img.to(device)

    if verbose:
        print(f'Content image shape: {content_img.shape}')
        print(f'Style image shape: {style_img.shape}')
        print(f'Device: {device}')
        print(f'Running style transfer for {num_steps} steps...')

    # Run style transfer
    output = run_style_transfer(
        content_img=content_img,
        style_img=style_img,
        num_steps=num_steps,
        style_weight=style_weight,
        content_weight=content_weight,
        content_layers=content_layers,
        style_layers=style_layers,
        init_image=init_image,
        verbose=verbose
    )

    # Convert to numpy array
    return tensor_to_image(output)


# Quick test / demo
if __name__ == '__main__':
    import os

    # test images
    dataset_path = 'style-transfer-dataset'
    content_path = os.path.join(dataset_path, 'contents', 'content_38.jpg')
    style_path = os.path.join(dataset_path, 'styles', 'style_18.jpg')

    if os.path.exists(content_path) and os.path.exists(style_path):
        print('Running style transfer demo...')

        # Using the simple function interface
        stylized = stylize_image(
            content_path=content_path,
            style_path=style_path,
            output_size=1280,  # Smaller for faster demo
            num_steps=500,
            style_weight=1e4,
            content_weight=3,
            verbose=True
        )

        # Save result
        output_path = 'stylized_output.jpg'
        Image.fromarray(stylized).save(output_path)
        print(f'Saved stylized image to {output_path}')
    else:
        print('Dataset not found. Please ensure style-transfer-dataset exists.')
        print(f'Looking for: {content_path}')
