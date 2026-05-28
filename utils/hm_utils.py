"""Demo-side image helpers (star marker + mask overlay).

Slim version: keeps only the functions imported by the GLaMM / Sa2VA demos.
The full hm_utils used at training time (S3/EXR loaders, ISP augmentation,
multi-channel readers, etc.) is not shipped in the release.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch


COLOR_MAP = {
    "red":       (1.0, 0.0, 0.0),
    "green":     (0.0, 1.0, 0.0),
    "blue":      (0.0, 0.0, 1.0),
    "cyan":      (0.0, 1.0, 1.0),
    "magenta":   (1.0, 0.0, 1.0),
    "yellow":    (1.0, 1.0, 0.0),
    "orange":    (1.0, 0.5, 0.0),
    "purple":    (0.5, 0.0, 1.0),
    "pink":      (1.0, 0.5, 0.8),
    "turquoise": (0.0, 0.8, 0.8),
}


def ensure_three_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[-1] == 1:
        return np.repeat(image, 3, axis=-1)
    return image


def select_contrasting_color(avg_color) -> Tuple[str, Tuple[float, float, float]]:
    """Pick the COLOR_MAP entry farthest (L2 in RGB) from ``avg_color``."""
    if isinstance(avg_color, torch.Tensor):
        avg_color = avg_color.detach().cpu().numpy()
    avg_color = np.asarray(avg_color, dtype=np.float32).reshape(-1)
    if avg_color.size < 3:
        avg_color = np.pad(avg_color, (0, 3 - avg_color.size), constant_values=0)

    best_contrast = -1.0
    best_name = "red"
    best_rgb = COLOR_MAP[best_name]
    for name, rgb in COLOR_MAP.items():
        contrast = sum((rgb[i] - float(avg_color[i])) ** 2 for i in range(3))
        if contrast > best_contrast:
            best_contrast = contrast
            best_name = name
            best_rgb = rgb
    return best_name, best_rgb


def create_mask_overlay(
    original: np.ndarray,
    mask_tensor: Optional[torch.Tensor],
    alpha: float = 0.45,
) -> Tuple[np.ndarray, str]:
    """Blend a binary mask onto a uint8 RGB image with an auto-contrast colour."""
    base_img = ensure_three_channel(original)
    if mask_tensor is None:
        return base_img.copy(), "red"

    mask_np = mask_tensor.detach().cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    mask_np = mask_np.astype(bool)

    normalized = base_img.astype(np.float32) / 255.0
    if mask_np.any():
        avg_color = normalized[mask_np].mean(axis=0)
    else:
        avg_color = normalized.mean(axis=(0, 1))

    color_name, color_rgb = select_contrasting_color(avg_color)
    color_arr = np.asarray(color_rgb, dtype=np.float32)

    blended = normalized.copy()
    blended[mask_np] = blended[mask_np] * (1 - alpha) + color_arr * alpha
    overlay_img = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return overlay_img, color_name


def augment_data(image, mat_label, size, flip=True, test=False, crop=True,
                 resize=False, point_hw=None):
    """Crop/flip augmentation on CHW float tensors.

    image: [C, H, W] float in [0, 1]
    mat_label: [N, H, W] float mask(s)
    size: int — output spatial size (square)
    """
    import random as _random
    import torchvision.transforms.functional as TF
    from torchvision.transforms.functional import InterpolationMode

    C, H, W = image.shape
    out_h, out_w = (size, size) if isinstance(size, int) else (size[0], size[1])

    if flip and not test and _random.random() < 0.5:
        image = TF.hflip(image)
        mat_label = TF.hflip(mat_label)

    if crop:
        if test or point_hw is None:
            top = max(0, (H - out_h) // 2)
            left = max(0, (W - out_w) // 2)
        else:
            ph, pw = point_hw
            top = max(0, min(H - out_h, ph - out_h // 2))
            left = max(0, min(W - out_w, pw - out_w // 2))
        image = TF.crop(image, top, left, min(out_h, H), min(out_w, W))
        mat_label = TF.crop(mat_label, top, left, min(out_h, H), min(out_w, W))

    cur_h, cur_w = image.shape[-2], image.shape[-1]
    if cur_h != out_h or cur_w != out_w:
        image = TF.resize(image, [out_h, out_w], antialias=True)
        mat_label = TF.resize(mat_label, [out_h, out_w], interpolation=InterpolationMode.NEAREST_EXACT)

    return image, mat_label


def add_star_marker(image, h, w, size=None, color=None):
    """Draw a filled 5-point star at (h, w) on a CHW float tensor in [0, 1].

    Args:
        image: torch.Tensor of shape [C, H, W].
        h, w: integer pixel centre (h = row, w = column).
        size: bounding-box size in pixels. Default 10.
        color: one of: None (auto-pick contrasting), a key in COLOR_MAP,
               or an (R, G, B) tuple / tensor in [0, 1].

    Returns:
        (marked_image, color_name).
    """
    marked_image = image.clone()
    C, H, W = image.shape

    if size is None:
        size = 10
    outer_radius = size // 2
    inner_radius = outer_radius * 0.4

    best_color_name = "red"
    color_rgb = COLOR_MAP[best_color_name]

    if color is None:
        sample_size = size
        y0, y1 = max(0, h - sample_size), min(H, h + sample_size)
        x0, x1 = max(0, w - sample_size), min(W, w + sample_size)
        nearby = image[:, y0:y1, x0:x1]
        if nearby.numel() > 0:
            avg_color = nearby.mean(dim=(1, 2))
            best_color_name, color_rgb = select_contrasting_color(avg_color)
    elif isinstance(color, str):
        if color in COLOR_MAP:
            best_color_name = color
            color_rgb = COLOR_MAP[color]
    else:
        best_color_name, color_rgb = select_contrasting_color(color)

    points = []
    for i in range(10):
        angle = i * np.pi / 5 - np.pi / 2
        r = outer_radius if i % 2 == 0 else inner_radius
        py = max(0, min(H - 1, h + int(r * np.sin(angle))))
        px = max(0, min(W - 1, w + int(r * np.cos(angle))))
        points.append((py, px))

    points_np = np.array([(p[1], p[0]) for p in points], dtype=np.int32)
    mask_np = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask_np, [points_np], 1)
    mask = torch.from_numpy(mask_np).bool().to(marked_image.device)

    color_tensor = torch.tensor(
        color_rgb, dtype=marked_image.dtype, device=marked_image.device
    ).view(3, 1)
    marked_image[:, mask] = color_tensor

    return marked_image, best_color_name
