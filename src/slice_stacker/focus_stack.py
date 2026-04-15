#!/usr/bin/env python3
"""
Focus stacking for 16-bit TIFF images from focus bracketing.
Memory-efficient: processes images incrementally, never loads all at once.

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


def load_image(path: Path) -> np.ndarray:
    """Load a single 16-bit TIFF image."""
    img = tifffile.imread(path)
    if img is None:
        raise ValueError(f"Failed to load: {path}")
    return img


def get_image_info(paths: list[Path]) -> tuple[tuple, np.dtype]:
    """Get shape and dtype from first image, validate all match."""
    first = tifffile.imread(paths[0])
    shape, dtype = first.shape, first.dtype
    print(f"Reference: {paths[0].name} shape={shape} dtype={dtype}")
    
    # Quick validation of remaining files (just headers, not full load)
    for p in paths[1:]:
        img = tifffile.imread(p)
        if img.shape != shape:
            raise ValueError(f"Shape mismatch: {p.name} is {img.shape}, expected {shape}")
        del img
    
    return shape, dtype


def compute_focus_measure(img: np.ndarray, method: str = "laplacian", 
                          kernel_size: int = 5) -> np.ndarray:
    """
    Compute per-pixel focus measure (sharpness map).
    Returns float32 to save memory vs float64.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    
    if method == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=kernel_size)
        focus = np.abs(lap)
    elif method == "gradient":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel_size)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=kernel_size)
        focus = np.sqrt(gx**2 + gy**2)
    elif method == "variance":
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        focus = sqr_mean - mean**2
    else:
        raise ValueError(f"Unknown focus method: {method}")
    
    return focus


def compute_warp_matrix(ref_gray: np.ndarray, img_gray: np.ndarray) -> np.ndarray | None:
    """Compute ECC warp matrix for alignment. Returns None on failure."""
    warp_mode = cv2.MOTION_EUCLIDEAN
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    try:
        _, warp_matrix = cv2.findTransformECC(ref_gray, img_gray, warp_matrix, warp_mode, criteria)
        return warp_matrix
    except cv2.error:
        return None


def apply_warp(img: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
    """Apply warp matrix to image."""
    h, w = img.shape[:2]
    return cv2.warpAffine(img, warp_matrix, (w, h),
                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def stack_max_streaming(paths: list[Path], focus_method: str, kernel_size: int,
                        smooth_radius: int, align: bool, ref_idx: int) -> np.ndarray:
    """
    Memory-efficient max-sharpness stacking.
    Pass 1: compute all focus measures, track best index per pixel.
    Pass 2: load images as needed to build output.
    """
    n = len(paths)
    shape, dtype = get_image_info(paths)
    h, w = shape[:2]
    is_color = len(shape) == 3
    
    # Prepare alignment reference if needed
    ref_gray = None
    warp_matrices = [None] * n
    if align:
        print(f"Loading reference image {ref_idx} for alignment...")
        ref_img = load_image(paths[ref_idx])
        if is_color:
            ref_gray = cv2.cvtColor((ref_img / 256).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            ref_gray = (ref_img / 256).astype(np.uint8)
        del ref_img
    
    # Pass 1: Compute focus measures, track best
    print(f"Pass 1: Computing focus measures ({focus_method})...")
    best_focus = np.zeros((h, w), dtype=np.float32)
    best_idx = np.zeros((h, w), dtype=np.uint16)
    
    smooth_ksize = smooth_radius * 2 + 1 if smooth_radius > 0 else 0
    
    for i, p in enumerate(paths):
        print(f"  [{i+1}/{n}] {p.name}")
        img = load_image(p)
        
        # Align if requested
        if align and i != ref_idx:
            if is_color:
                img_gray = cv2.cvtColor((img / 256).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                img_gray = (img / 256).astype(np.uint8)
            warp_matrices[i] = compute_warp_matrix(ref_gray, img_gray)
            if warp_matrices[i] is not None:
                img = apply_warp(img, warp_matrices[i])
        
        focus = compute_focus_measure(img, focus_method, kernel_size)
        if smooth_ksize > 0:
            focus = cv2.GaussianBlur(focus, (smooth_ksize, smooth_ksize), 0)
        
        # Update best
        better = focus > best_focus
        best_focus = np.where(better, focus, best_focus)
        best_idx = np.where(better, i, best_idx).astype(np.uint16)
        
        del img, focus
    
    del best_focus  # No longer needed
    
    # Pass 2: Build output by loading images as needed
    print("Pass 2: Assembling output...")
    
    # Find which images are actually used
    used_indices = np.unique(best_idx)
    print(f"  {len(used_indices)} images contribute to output")
    
    if is_color:
        result = np.zeros((h, w, shape[2]), dtype=dtype)
    else:
        result = np.zeros((h, w), dtype=dtype)
    
    for i in used_indices:
        mask = best_idx == i
        if not np.any(mask):
            continue
        
        print(f"  Loading {paths[i].name} for {np.sum(mask)} pixels")
        img = load_image(paths[i])
        
        if align and warp_matrices[i] is not None:
            img = apply_warp(img, warp_matrices[i])
        
        if is_color:
            for c in range(shape[2]):
                result[:, :, c] = np.where(mask, img[:, :, c], result[:, :, c])
        else:
            result = np.where(mask, img, result)
        
        del img
    
    return result


def stack_weighted_streaming(paths: list[Path], focus_method: str, kernel_size: int,
                             smooth_radius: int, align: bool, ref_idx: int) -> np.ndarray:
    """
    Memory-efficient weighted average stacking.
    Pass 1: compute all focus measures, accumulate sum.
    Pass 2: normalize and blend.
    """
    n = len(paths)
    shape, dtype = get_image_info(paths)
    h, w = shape[:2]
    is_color = len(shape) == 3
    
    # Prepare alignment
    ref_gray = None
    warp_matrices = [None] * n
    if align:
        print(f"Loading reference image {ref_idx} for alignment...")
        ref_img = load_image(paths[ref_idx])
        if is_color:
            ref_gray = cv2.cvtColor((ref_img / 256).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            ref_gray = (ref_img / 256).astype(np.uint8)
        del ref_img
    
    smooth_ksize = smooth_radius * 2 + 1 if smooth_radius > 0 else 0
    
    # Pass 1: Compute focus sum
    print(f"Pass 1: Computing focus measures ({focus_method})...")
    focus_sum = np.zeros((h, w), dtype=np.float64)
    focus_maps = []  # Store paths and will recompute in pass 2
    
    for i, p in enumerate(paths):
        print(f"  [{i+1}/{n}] {p.name}")
        img = load_image(p)
        
        if align and i != ref_idx:
            if is_color:
                img_gray = cv2.cvtColor((img / 256).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                img_gray = (img / 256).astype(np.uint8)
            warp_matrices[i] = compute_warp_matrix(ref_gray, img_gray)
            if warp_matrices[i] is not None:
                img = apply_warp(img, warp_matrices[i])
        
        focus = compute_focus_measure(img, focus_method, kernel_size)
        if smooth_ksize > 0:
            focus = cv2.GaussianBlur(focus, (smooth_ksize, smooth_ksize), 0)
        
        focus_sum += focus.astype(np.float64)
        del img, focus
    
    focus_sum = np.maximum(focus_sum, 1e-10)
    
    # Pass 2: Weighted accumulation
    print("Pass 2: Blending...")
    if is_color:
        result = np.zeros((h, w, shape[2]), dtype=np.float64)
    else:
        result = np.zeros((h, w), dtype=np.float64)
    
    for i, p in enumerate(paths):
        print(f"  [{i+1}/{n}] {p.name}")
        img = load_image(p)
        
        if align and warp_matrices[i] is not None:
            img = apply_warp(img, warp_matrices[i])
        
        focus = compute_focus_measure(img, focus_method, kernel_size)
        if smooth_ksize > 0:
            focus = cv2.GaussianBlur(focus, (smooth_ksize, smooth_ksize), 0)
        
        weight = focus.astype(np.float64) / focus_sum
        
        if is_color:
            for c in range(shape[2]):
                result[:, :, c] += img[:, :, c].astype(np.float64) * weight
        else:
            result += img.astype(np.float64) * weight
        
        del img, focus, weight
    
    return np.clip(result, 0, 65535).astype(np.uint16)


def stack_pyramid_streaming(paths: list[Path], focus_method: str, kernel_size: int,
                            levels: int, align: bool, ref_idx: int) -> np.ndarray:
    """
    Memory-efficient Laplacian pyramid blending.
    Builds blended pyramid incrementally instead of storing all pyramids.
    """
    n = len(paths)
    shape, dtype = get_image_info(paths)
    h, w = shape[:2]
    is_color = len(shape) == 3
    channels = shape[2] if is_color else 1
    
    # Prepare alignment
    ref_gray = None
    warp_matrices = [None] * n
    if align:
        print(f"Loading reference image {ref_idx} for alignment...")
        ref_img = load_image(paths[ref_idx])
        if is_color:
            ref_gray = cv2.cvtColor((ref_img / 256).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            ref_gray = (ref_img / 256).astype(np.uint8)
        del ref_img
    
    # Pass 1: Compute focus sum for normalization
    print(f"Pass 1: Computing focus measures ({focus_method})...")
    focus_sum = np.zeros((h, w), dtype=np.float64)
    
    for i, p in enumerate(paths):
        print(f"  [{i+1}/{n}] {p.name}")
        img = load_image(p)
        
        if align and i != ref_idx:
            if is_color:
                img_gray = cv2.cvtColor((img / 256).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                img_gray = (img / 256).astype(np.uint8)
            warp_matrices[i] = compute_warp_matrix(ref_gray, img_gray)
            if warp_matrices[i] is not None:
                img = apply_warp(img, warp_matrices[i])
        
        focus = compute_focus_measure(img, focus_method, kernel_size)
        focus = cv2.GaussianBlur(focus, (31, 31), 0)
        focus_sum += focus.astype(np.float64)
        del img, focus
    
    focus_sum = np.maximum(focus_sum, 1e-10)
    
    # Build focus_sum pyramid for normalization at each level
    focus_sum_pyr = _build_gaussian_pyramid(focus_sum.astype(np.float32), levels)
    
    # Initialize blended pyramid accumulators (per channel)
    print(f"Pass 2: Building blended pyramid ({levels} levels)...")
    blended_pyramids = []  # One per channel
    for _ in range(channels):
        pyr = []
        current_h, current_w = h, w
        for lv in range(levels - 1):
            pyr.append(np.zeros((current_h, current_w), dtype=np.float64))
            current_h = (current_h + 1) // 2
            current_w = (current_w + 1) // 2
        pyr.append(np.zeros((current_h, current_w), dtype=np.float64))
        blended_pyramids.append(pyr)
    
    # Weight sum pyramid for normalization
    weight_sum_pyramids = []
    for _ in range(channels):
        pyr = []
        current_h, current_w = h, w
        for lv in range(levels - 1):
            pyr.append(np.zeros((current_h, current_w), dtype=np.float64))
            current_h = (current_h + 1) // 2
            current_w = (current_w + 1) // 2
        pyr.append(np.zeros((current_h, current_w), dtype=np.float64))
        weight_sum_pyramids.append(pyr)
    
    # Accumulate weighted pyramids
    for i, p in enumerate(paths):
        print(f"  [{i+1}/{n}] {p.name}")
        img = load_image(p)
        
        if align and warp_matrices[i] is not None:
            img = apply_warp(img, warp_matrices[i])
        
        # Compute weight
        focus = compute_focus_measure(img, focus_method, kernel_size)
        focus = cv2.GaussianBlur(focus, (31, 31), 0)
        weight = focus.astype(np.float64) / focus_sum
        weight_pyr = _build_gaussian_pyramid(weight.astype(np.float32), levels)
        
        # Process each channel
        for c in range(channels):
            if is_color:
                channel = img[:, :, c].astype(np.float32)
            else:
                channel = img.astype(np.float32)
            
            img_lap_pyr = _build_laplacian_pyramid(channel, levels)
            
            for lv in range(levels):
                blended_pyramids[c][lv] += img_lap_pyr[lv].astype(np.float64) * weight_pyr[lv]
                weight_sum_pyramids[c][lv] += weight_pyr[lv]
        
        del img, focus, weight, weight_pyr
    
    # Normalize and collapse
    print("Collapsing pyramid...")
    result_channels = []
    for c in range(channels):
        for lv in range(levels):
            weight_sum_pyramids[c][lv] = np.maximum(weight_sum_pyramids[c][lv], 1e-10)
            blended_pyramids[c][lv] /= weight_sum_pyramids[c][lv]
        
        collapsed = _collapse_laplacian_pyramid(blended_pyramids[c])
        result_channels.append(collapsed)
    
    if is_color:
        result = np.stack(result_channels, axis=2)
    else:
        result = result_channels[0]
    
    return np.clip(result, 0, 65535).astype(np.uint16)


def _build_laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build Laplacian pyramid."""
    pyramid = []
    current = img.astype(np.float32)
    
    for _ in range(levels - 1):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        lap = current - up
        pyramid.append(lap)
        current = down
    
    pyramid.append(current)
    return pyramid


def _build_gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build Gaussian pyramid."""
    pyramid = [img.astype(np.float32)]
    current = img.astype(np.float32)
    
    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid


def _collapse_laplacian_pyramid(pyramid: list[np.ndarray]) -> np.ndarray:
    """Reconstruct image from Laplacian pyramid."""
    current = pyramid[-1].astype(np.float64)
    
    for i in range(len(pyramid) - 2, -1, -1):
        up = cv2.pyrUp(current.astype(np.float32), dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        current = up.astype(np.float64) + pyramid[i].astype(np.float64)
    
    return current


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
    
    n_images = len(args.images)
    print(f"Processing {n_images} images")
    
    # Determine alignment reference
    ref_idx = args.align_ref if args.align_ref >= 0 else n_images // 2
    
    # Stack using streaming methods
    print(f"Stacking with method: {args.method}")
    if args.method == "max":
        result = stack_max_streaming(args.images, args.focus_measure, args.kernel_size,
                                     args.smooth_weights, args.align, ref_idx)
    elif args.method == "weighted":
        result = stack_weighted_streaming(args.images, args.focus_measure, args.kernel_size,
                                          args.smooth_weights, args.align, ref_idx)
    elif args.method == "pyramid":
        result = stack_pyramid_streaming(args.images, args.focus_measure, args.kernel_size,
                                         args.pyramid_levels, args.align, ref_idx)
    
    # Save result
    tifffile.imwrite(args.output, result, photometric='rgb' if result.ndim == 3 else 'minisblack')
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
