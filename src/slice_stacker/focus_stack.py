#!/usr/bin/env python3
"""
Focus stacking for 16-bit TIFF images from focus bracketing.

Usage:
    python focus_stack.py image1.tif image2.tif ... -o output.tif
    python focus_stack.py *.tif -o stacked.tif --method pyramid --align
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile


def load_images(paths: list[Path]) -> list[np.ndarray]:
    """Load 16-bit TIFF images, preserving bit depth."""
    images = []
    for p in paths:
        img = tifffile.imread(p)
        if img is None:
            raise ValueError(f"Failed to load: {p}")
        # Ensure we have a consistent shape (H, W, C) for color or (H, W) for mono
        images.append(img)
        print(f"Loaded: {p.name} shape={img.shape} dtype={img.dtype}")
    
    # Validate all images have same shape
    shapes = [img.shape for img in images]
    if len(set(shapes)) > 1:
        raise ValueError(f"Image shape mismatch: {shapes}")
    
    return images


def align_images(images: list[np.ndarray], reference_idx: int = 0) -> list[np.ndarray]:
    """
    Align images using ECC (Enhanced Correlation Coefficient).
    Works well for small translations/rotations typical in focus brackets.
    """
    print(f"Aligning {len(images)} images to reference index {reference_idx}...")
    
    ref = images[reference_idx]
    # Convert to 8-bit grayscale for alignment computation
    if ref.ndim == 3:
        ref_gray = cv2.cvtColor((ref / 256).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        ref_gray = (ref / 256).astype(np.uint8)
    
    aligned = []
    warp_mode = cv2.MOTION_EUCLIDEAN
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
    
    for i, img in enumerate(images):
        if i == reference_idx:
            aligned.append(img)
            continue
        
        if img.ndim == 3:
            img_gray = cv2.cvtColor((img / 256).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            img_gray = (img / 256).astype(np.uint8)
        
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp_matrix = cv2.findTransformECC(ref_gray, img_gray, warp_matrix, warp_mode, criteria)
            
            h, w = img.shape[:2]
            if img.ndim == 3:
                aligned_img = cv2.warpAffine(img, warp_matrix, (w, h),
                                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                             borderMode=cv2.BORDER_REFLECT)
            else:
                aligned_img = cv2.warpAffine(img, warp_matrix, (w, h),
                                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                             borderMode=cv2.BORDER_REFLECT)
            aligned.append(aligned_img)
            print(f"  Aligned image {i}: dx={warp_matrix[0,2]:.2f} dy={warp_matrix[1,2]:.2f}")
        except cv2.error as e:
            print(f"  Warning: ECC failed for image {i}, using unaligned: {e}")
            aligned.append(img)
    
    return aligned


def compute_focus_measure(img: np.ndarray, method: str = "laplacian", 
                          kernel_size: int = 5) -> np.ndarray:
    """
    Compute per-pixel focus measure (sharpness map).
    
    Methods:
        laplacian: Absolute Laplacian response (good general choice)
        gradient: Gradient magnitude (Sobel)
        variance: Local variance (good for texture)
    """
    # Work in float64 for precision with 16-bit data
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        gray = img.astype(np.float64)
    
    if method == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        focus = np.abs(lap)
    elif method == "gradient":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        focus = np.sqrt(gx**2 + gy**2)
    elif method == "variance":
        # Local variance using box filter
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        focus = sqr_mean - mean**2
    else:
        raise ValueError(f"Unknown focus method: {method}")
    
    return focus


def stack_max_sharpness(images: list[np.ndarray], focus_method: str = "laplacian",
                        kernel_size: int = 5, smooth_weights: int = 0) -> np.ndarray:
    """
    Stack by selecting the sharpest pixel from each image.
    Fast and effective for well-aligned stacks.
    """
    print(f"Computing focus measures ({focus_method})...")
    focus_maps = [compute_focus_measure(img, focus_method, kernel_size) for img in images]
    
    # Stack focus maps: shape (N, H, W)
    focus_stack = np.stack(focus_maps, axis=0)
    
    # Optional: smooth the focus maps to reduce noise in selection
    if smooth_weights > 0:
        for i in range(len(focus_maps)):
            focus_stack[i] = cv2.GaussianBlur(focus_stack[i], 
                                               (smooth_weights*2+1, smooth_weights*2+1), 0)
    
    # Find index of maximum focus at each pixel
    best_idx = np.argmax(focus_stack, axis=0)
    
    # Build output by selecting from the best image at each pixel
    img_stack = np.stack(images, axis=0)
    h, w = best_idx.shape
    
    if images[0].ndim == 3:
        c = images[0].shape[2]
        result = np.zeros((h, w, c), dtype=images[0].dtype)
        for ch in range(c):
            result[:, :, ch] = np.take_along_axis(
                img_stack[:, :, :, ch], best_idx[np.newaxis, :, :], axis=0)[0]
    else:
        result = np.take_along_axis(img_stack, best_idx[np.newaxis, :, :], axis=0)[0]
    
    return result


def stack_weighted(images: list[np.ndarray], focus_method: str = "laplacian",
                   kernel_size: int = 5, smooth_weights: int = 11) -> np.ndarray:
    """
    Stack using weighted average based on focus measure.
    Smoother transitions than max selection.
    """
    print(f"Computing focus measures ({focus_method})...")
    focus_maps = [compute_focus_measure(img, focus_method, kernel_size) for img in images]
    
    # Smooth focus maps for blending weights
    if smooth_weights > 0:
        focus_maps = [cv2.GaussianBlur(fm, (smooth_weights*2+1, smooth_weights*2+1), 0) 
                      for fm in focus_maps]
    
    # Stack and normalize to create weights
    focus_stack = np.stack(focus_maps, axis=0)
    weight_sum = np.sum(focus_stack, axis=0, keepdims=True)
    weight_sum = np.maximum(weight_sum, 1e-10)  # avoid division by zero
    weights = focus_stack / weight_sum
    
    # Weighted average
    img_stack = np.stack(images, axis=0).astype(np.float64)
    
    if images[0].ndim == 3:
        weights = weights[:, :, :, np.newaxis]
    
    result = np.sum(img_stack * weights, axis=0)
    return np.clip(result, 0, 65535).astype(np.uint16)


def build_laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build Laplacian pyramid for multi-scale blending."""
    pyramid = []
    current = img.astype(np.float64)
    
    for _ in range(levels - 1):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        lap = current - up
        pyramid.append(lap)
        current = down
    
    pyramid.append(current)  # Residual at coarsest level
    return pyramid


def build_gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build Gaussian pyramid for weight maps."""
    pyramid = [img.astype(np.float64)]
    current = img.astype(np.float64)
    
    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid


def collapse_laplacian_pyramid(pyramid: list[np.ndarray]) -> np.ndarray:
    """Reconstruct image from Laplacian pyramid."""
    current = pyramid[-1]
    
    for i in range(len(pyramid) - 2, -1, -1):
        up = cv2.pyrUp(current, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        current = up + pyramid[i]
    
    return current


def stack_pyramid(images: list[np.ndarray], focus_method: str = "laplacian",
                  kernel_size: int = 5, levels: int = 6) -> np.ndarray:
    """
    Laplacian pyramid blending based on focus measure.
    Best quality, handles depth-of-field transitions smoothly.
    """
    print(f"Computing focus measures ({focus_method})...")
    focus_maps = [compute_focus_measure(img, focus_method, kernel_size) for img in images]
    
    # Smooth focus maps
    focus_maps = [cv2.GaussianBlur(fm, (31, 31), 0) for fm in focus_maps]
    
    # Normalize to weights
    focus_stack = np.stack(focus_maps, axis=0)
    weight_sum = np.sum(focus_stack, axis=0, keepdims=True)
    weight_sum = np.maximum(weight_sum, 1e-10)
    weights = focus_stack / weight_sum
    
    print(f"Building pyramids ({levels} levels)...")
    
    # Process each channel separately for color images
    is_color = images[0].ndim == 3
    if is_color:
        channels = images[0].shape[2]
        result_channels = []
        
        for ch in range(channels):
            channel_imgs = [img[:, :, ch] for img in images]
            blended = _blend_channel_pyramid(channel_imgs, weights, levels)
            result_channels.append(blended)
        
        result = np.stack(result_channels, axis=2)
    else:
        result = _blend_channel_pyramid(images, weights, levels)
    
    return np.clip(result, 0, 65535).astype(np.uint16)


def _blend_channel_pyramid(images: list[np.ndarray], weights: np.ndarray, 
                           levels: int) -> np.ndarray:
    """Blend single channel using Laplacian pyramid."""
    n_images = len(images)
    
    # Build pyramids for each image and weight
    img_pyramids = [build_laplacian_pyramid(img, levels) for img in images]
    weight_pyramids = [build_gaussian_pyramid(w, levels) for w in weights]
    
    # Blend at each pyramid level
    blended_pyramid = []
    for level in range(levels):
        blended_level = np.zeros_like(img_pyramids[0][level])
        weight_sum = np.zeros_like(weight_pyramids[0][level])
        
        for i in range(n_images):
            blended_level += img_pyramids[i][level] * weight_pyramids[i][level]
            weight_sum += weight_pyramids[i][level]
        
        weight_sum = np.maximum(weight_sum, 1e-10)
        blended_pyramid.append(blended_level / weight_sum)
    
    return collapse_laplacian_pyramid(blended_pyramid)


def main():
    parser = argparse.ArgumentParser(
        description="Focus stack 16-bit TIFF images from focus bracketing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s IMG_*.tif -o stacked.tif
  %(prog)s *.tif -o out.tif --method pyramid --align
  %(prog)s img1.tif img2.tif img3.tif -o result.tif --focus-measure gradient

Stacking methods:
  max      - Select sharpest pixel from each image (fast, can be harsh)
  weighted - Weighted average by focus measure (smoother)
  pyramid  - Laplacian pyramid blending (best quality, slower)

Focus measures:
  laplacian - Laplacian response (good default)
  gradient  - Sobel gradient magnitude
  variance  - Local variance (good for texture)
        """
    )
    
    parser.add_argument("images", nargs="+", type=Path,
                        help="Input 16-bit TIFF files")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output TIFF filename")
    parser.add_argument("--method", choices=["max", "weighted", "pyramid"],
                        default="pyramid",
                        help="Stacking method (default: pyramid)")
    parser.add_argument("--focus-measure", choices=["laplacian", "gradient", "variance"],
                        default="laplacian", dest="focus_measure",
                        help="Focus measure algorithm (default: laplacian)")
    parser.add_argument("--kernel-size", type=int, default=5, dest="kernel_size",
                        help="Kernel size for focus measure (default: 5)")
    parser.add_argument("--align", action="store_true",
                        help="Align images before stacking (ECC algorithm)")
    parser.add_argument("--align-ref", type=int, default=-1, dest="align_ref",
                        help="Reference image index for alignment (default: middle)")
    parser.add_argument("--pyramid-levels", type=int, default=6, dest="pyramid_levels",
                        help="Pyramid levels for pyramid method (default: 6)")
    parser.add_argument("--smooth-weights", type=int, default=11, dest="smooth_weights",
                        help="Gaussian smoothing radius for weight maps (default: 11)")
    
    args = parser.parse_args()
    
    if len(args.images) < 2:
        parser.error("Need at least 2 images to stack")
    
    # Verify all input files exist
    for p in args.images:
        if not p.exists():
            parser.error(f"File not found: {p}")
    
    # Load images
    images = load_images(args.images)
    print(f"Loaded {len(images)} images")
    
    # Optionally align
    if args.align:
        ref_idx = args.align_ref if args.align_ref >= 0 else len(images) // 2
        images = align_images(images, reference_idx=ref_idx)
    
    # Stack
    print(f"Stacking with method: {args.method}")
    if args.method == "max":
        result = stack_max_sharpness(images, args.focus_measure, args.kernel_size,
                                     args.smooth_weights)
    elif args.method == "weighted":
        result = stack_weighted(images, args.focus_measure, args.kernel_size,
                                args.smooth_weights)
    elif args.method == "pyramid":
        result = stack_pyramid(images, args.focus_measure, args.kernel_size,
                               args.pyramid_levels)
    
    # Save result
    tifffile.imwrite(args.output, result, photometric='rgb' if result.ndim == 3 else 'minisblack')
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
