"""
Generate patches along a contour at a specified distance from tissue edge.

This approach samples from the "sweet spot" region of whole slide images,
avoiding both the outer edge (artifacts, folding) and the center.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import openslide


def get_tissue_mask(slide, target_mpp: float = 16.0, verbose: bool = True):
    """
    Generate a binary tissue mask from a whole slide image.

    Uses LAB color space for robust H&E tissue detection, which works reliably
    on both white and gray backgrounds. The 'a' channel separates pink/purple
    tissue from neutral gray/white backgrounds.

    Args:
        slide: OpenSlide object
        target_mpp: Target microns per pixel for thumbnail (higher = faster, lower = more detail)
        verbose: Print progress info

    Returns:
        mask: Binary numpy array (255 = tissue, 0 = background)
        downsample: Downsample factor from level 0
        actual_mpp: Actual microns per pixel of the mask
    """
    # Get slide mpp
    try:
        slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    except (KeyError, ValueError):
        slide_mpp = 0.25  # Default assumption for 40x slides
        if verbose:
            print(f"  Warning: MPP not found, assuming {slide_mpp} μm/px")

    # Find appropriate level (closest to target_mpp without exceeding it too much)
    target_downsample = target_mpp / slide_mpp
    best_level = 0

    for level in range(slide.level_count):
        if slide.level_downsamples[level] <= target_downsample * 1.5:
            best_level = level

    # Get thumbnail at this level
    level_dims = slide.level_dimensions[best_level]
    downsample = slide.level_downsamples[best_level]
    actual_mpp = slide_mpp * downsample

    if verbose:
        print(f"  Using level {best_level}: {level_dims[0]}x{level_dims[1]}, {actual_mpp:.2f} μm/px")

    # Read entire level as thumbnail
    thumbnail = slide.read_region((0, 0), best_level, level_dims)
    thumbnail_rgb = np.array(thumbnail.convert('RGB'))

    # === Primary method: LAB color space (robust for H&E on any background) ===
    # LAB 'a' channel: gray/white background ≈ 128, pink tissue > 128
    thumbnail_lab = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2LAB)
    l_channel = thumbnail_lab[:, :, 0]  # Lightness
    a_channel = thumbnail_lab[:, :, 1]  # Green-Red (pink tissue is high)
    b_channel = thumbnail_lab[:, :, 2]  # Blue-Yellow

    # H&E tissue detection:
    # - 'a' channel > 128 indicates pink/red/purple shift (eosin + hematoxylin)
    # - Lightness not too dark (avoid pen marks) and not too bright (avoid pure white)
    # - Use Otsu's method on 'a' channel for adaptive thresholding

    # Otsu's threshold on 'a' channel for robust separation
    _, a_mask_otsu = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Also require minimum 'a' value (tissue should be pink-shifted, a > 128)
    # and reasonable lightness (not too dark, not pure white)
    a_thresh = 133  # Slightly above neutral (128) to catch pink tissue
    l_low = 20      # Not too dark (pen marks, dust)
    l_high = 240    # Not overexposed white

    # Combine Otsu result with explicit thresholds for robustness
    mask_lab = (
        (a_channel > a_thresh) &
        (l_channel > l_low) &
        (l_channel < l_high)
    ).astype(np.uint8) * 255

    # Use the more permissive of Otsu and explicit threshold
    # (Otsu adapts to the specific slide, explicit catches edge cases)
    mask = cv2.bitwise_or(mask_lab, a_mask_otsu)

    # === Fallback: also check saturation for slides with unusual staining ===
    thumbnail_hsv = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2HSV)
    saturation = thumbnail_hsv[:, :, 1]
    value = thumbnail_hsv[:, :, 2]

    # High saturation indicates colored tissue (works for non-H&E too)
    sat_thresh = 25  # Conservative threshold
    mask_hsv = (
        (saturation > sat_thresh) &
        (value > 30) &
        (value < 245)
    ).astype(np.uint8) * 255

    # Combine LAB and HSV masks (union)
    mask = cv2.bitwise_or(mask, mask_hsv)

    # Morphological cleanup
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # Close small holes within tissue
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
    # Remove small noise/debris
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

    # Keep only significant connected components
    # This removes isolated small tissue fragments and noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels > 1:
        # Find areas of each component (excluding background at index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) > 0:
            # Keep components larger than 1% of the largest
            max_area = areas.max()
            min_area_thresh = max_area * 0.01

            clean_mask = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area_thresh:
                    clean_mask[labels == i] = 255
            mask = clean_mask

    if verbose:
        tissue_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        tissue_percent = 100 * tissue_pixels / total_pixels
        print(f"  Tissue coverage: {tissue_percent:.1f}%")

    return mask, downsample, actual_mpp


def get_contour_at_distance(
    mask,
    mpp: float,
    distance_um: float | None = None,
    distance_fraction: float | None = None,
    verbose: bool = True,
):
    """
    Get contours at a specified distance from tissue edge using distance transform.

    Supports two modes:
    - Absolute: same distance (in μm) for all tissues
    - Fractional: distance as fraction of each tissue's diameter (per-tissue)

    Args:
        mask: Binary tissue mask
        mpp: Microns per pixel of the mask
        distance_um: Target distance from edge in microns (absolute mode)
        distance_fraction: Target distance as fraction of diameter, 0-0.5 (fractional mode)
        verbose: Print progress info

    Returns:
        contours: List of OpenCV contours at the target distance(s)
        dist_transform: Distance transform array (for debugging/visualization)
        mask_filled: Binary mask with internal holes filled
    """
    # Validate: exactly one mode must be specified
    if (distance_um is None) == (distance_fraction is None):
        raise ValueError("Exactly one of distance_um or distance_fraction must be specified")

    # Fill internal holes before computing distance transform.
    # This ensures we measure distance from the OUTER boundary only,
    # not from internal voids, tears, or tubular lumens within the tissue.
    external_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    cv2.drawContours(mask_filled, external_contours, -1, 255, cv2.FILLED)

    if verbose:
        holes_filled = np.sum((mask_filled > 0) & (mask == 0))
        if holes_filled > 0:
            hole_percent = 100 * holes_filled / np.sum(mask_filled > 0)
            print(f"  Filled internal holes: {hole_percent:.1f}% of tissue area")

    # Compute distance transform (Euclidean distance from OUTER boundary)
    dist_transform = cv2.distanceTransform(mask_filled, cv2.DIST_L2, 5)

    if distance_um is not None:
        # === ABSOLUTE MODE: same distance for all tissues ===
        distance_px = distance_um / mpp
        max_dist_px = dist_transform.max()
        max_dist_um = max_dist_px * mpp

        if verbose:
            print(f"  Max tissue depth: {max_dist_um:.0f} μm")
            print(f"  Target distance: {distance_um} μm ({distance_px:.1f} px)")

        if distance_px > max_dist_px:
            print(f"  Warning: Target distance {distance_um}μm exceeds max tissue depth {max_dist_um:.0f}μm")
            distance_px = max_dist_px * 0.8
            if verbose:
                print(f"  Using fallback distance: {distance_px * mpp:.0f} μm")

        # Threshold: everything at >= target distance
        inner_mask = (dist_transform >= distance_px).astype(np.uint8) * 255

        # The contour of this mask IS the isoline at target distance
        contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if verbose:
            print(f"  Found {len(contours)} contour(s)")

    else:
        # === FRACTIONAL MODE: per-tissue distance based on diameter ===
        if verbose:
            print(f"  Using fractional distance: {distance_fraction:.1%} of each tissue's diameter")

        # Find connected components (each tissue is a separate component)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels <= 1:
            # No tissue found (only background)
            return [], dist_transform

        all_contours = []

        for label_id in range(1, num_labels):
            # Find max distance (radius) within this tissue
            tissue_region = (labels == label_id)
            tissue_max_radius_px = dist_transform[tissue_region].max()
            tissue_diameter_um = 2 * tissue_max_radius_px * mpp

            # Calculate distance for this tissue: fraction × diameter
            tissue_distance_px = distance_fraction * 2 * tissue_max_radius_px

            if verbose:
                tissue_distance_um = tissue_distance_px * mpp
                print(f"    Tissue region {label_id}: {tissue_distance_um:.0f} μm from edge")

            # Skip if tissue is too small (would result in no contour)
            if tissue_distance_px >= tissue_max_radius_px:
                if verbose:
                    print(f"      (skipped - tissue too small for this fraction)")
                continue

            # Create mask for this tissue's inner region at its specific distance
            tissue_inner = (
                tissue_region & (dist_transform >= tissue_distance_px)
            ).astype(np.uint8) * 255

            # Find contours for this tissue
            tissue_contours, _ = cv2.findContours(
                tissue_inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            all_contours.extend(tissue_contours)

        contours = all_contours

        if verbose:
            print(f"  Found {len(contours)} contour(s) total")

    return contours, dist_transform, mask_filled


def get_contour_centroid(contour):
    """Calculate the centroid of a contour."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        # Fallback to mean of points if area is zero
        points = contour.reshape(-1, 2)
        return np.mean(points[:, 0]), np.mean(points[:, 1])
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx, cy


def sort_and_label_contours(contours, min_area_ratio: float = 0.1, verbose: bool = True):
    """
    Filter tiny fragments and sort contours by orientation-aware position.

    Step 1: Remove contours with area < min_area_ratio of the largest contour.
    Step 2: Detect orientation (horizontal vs vertical) based on centroid spread.
    Step 3: Sort by X (left→right) if horizontal, by Y (top→bottom) if vertical.

    Args:
        contours: List of OpenCV contours
        min_area_ratio: Minimum area as fraction of largest (default 0.1 = 10%)
        verbose: Print progress info

    Returns:
        List of (tissue_id, contour) tuples, sorted by position.
        tissue_id is 1-indexed.
    """
    if not contours:
        return []

    # Calculate areas and centroids for all contours
    contour_data = []
    for contour in contours:
        area = cv2.contourArea(contour)
        cx, cy = get_contour_centroid(contour)
        contour_data.append((contour, area, cx, cy))

    # Step 1: Filter tiny fragments
    max_area = max(d[1] for d in contour_data)
    min_area = max_area * min_area_ratio
    filtered = [d for d in contour_data if d[1] >= min_area]

    if verbose and len(filtered) < len(contour_data):
        removed = len(contour_data) - len(filtered)
        print(f"  Filtered out {removed} tiny fragment(s) (< {min_area_ratio:.0%} of largest)")

    # Step 2: Detect orientation and sort
    orientation = "single tissue"
    if len(filtered) > 1:
        x_coords = [d[2] for d in filtered]
        y_coords = [d[3] for d in filtered]
        x_spread = max(x_coords) - min(x_coords)
        y_spread = max(y_coords) - min(y_coords)

        if x_spread > y_spread:
            # Horizontal layout - sort left to right
            filtered.sort(key=lambda d: d[2])
            orientation = "horizontal (left→right)"
        else:
            # Vertical layout - sort top to bottom
            filtered.sort(key=lambda d: d[3])
            orientation = "vertical (top→bottom)"

    # Assign tissue IDs (1-indexed)
    labeled = [(tid, d[0]) for tid, d in enumerate(filtered, start=1)]

    if verbose:
        print(f"  Found {len(labeled)} tissue(s), {orientation}")

    return labeled


def sample_contour_points(labeled_contours, arc_step_px: float, verbose: bool = True):
    """
    Sample points along contours at regular arc-length intervals.

    Uses interpolation to place samples at exact arc-length positions,
    ensuring consistent spacing regardless of contour vertex density.

    Args:
        labeled_contours: List of (tissue_id, contour) tuples from sort_and_label_contours()
        arc_step_px: Distance between samples in pixels
        verbose: Print progress info

    Returns:
        List of (x, y, tissue_id) tuples in mask coordinates.
    """
    if not labeled_contours:
        return []

    all_points = []
    tissue_patch_counts = []  # Track patches per tissue for reporting

    for tissue_id, contour in labeled_contours:
        if len(contour) < 2:
            tissue_patch_counts.append((tissue_id, 0))
            continue

        points = contour.reshape(-1, 2).astype(np.float64)

        # Always include first point
        sampled = [(points[0][0], points[0][1], tissue_id)]
        accumulated = 0.0

        for i in range(1, len(points)):
            p_prev = points[i - 1]
            p_curr = points[i]

            dx = p_curr[0] - p_prev[0]
            dy = p_curr[1] - p_prev[1]
            segment_length = np.sqrt(dx * dx + dy * dy)

            if segment_length == 0:
                continue

            # Check if we can fit one or more samples in this segment
            accumulated += segment_length

            while accumulated >= arc_step_px:
                # Calculate interpolation factor from last sampled point
                # How far along from p_prev to p_curr should we sample?
                overshoot = accumulated - arc_step_px
                dist_from_prev = segment_length - overshoot
                t = dist_from_prev / segment_length

                # Interpolate the sample point
                sample_x = p_prev[0] + t * dx
                sample_y = p_prev[1] + t * dy
                sampled.append((sample_x, sample_y, tissue_id))

                accumulated -= arc_step_px

        tissue_patch_counts.append((tissue_id, len(sampled)))
        all_points.extend(sampled)

    if verbose:
        total_contours = [c for _, c in labeled_contours]
        total_length = sum(cv2.arcLength(c, closed=True) for c in total_contours)
        print(f"  Total contour length: {total_length:.0f} px")
        print(f"  Arc step: {arc_step_px:.1f} px")
        for tid, count in tissue_patch_counts:
            print(f"    Tissue {tid}: {count} patches")
        print(f"  Total: {len(all_points)} patch locations")

    return all_points


def save_debug_visualization(
    mask,
    dist_transform,
    contours,
    sample_points,
    output_path: str,
    distance_label,
):
    """
    Save a debug image showing tissue mask, distance contour, and sample points.

    Sample points are expected to be (x, y, tissue_id) tuples.
    Different tissues are shown in different colors.

    Args:
        distance_label: Either a number (μm) or a string (e.g., "25%") for display
    """
    # Create BGR visualization (OpenCV's native format)
    vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Show tissue mask in gray
    vis[mask > 0] = [100, 100, 100]

    # Normalize distance transform for visualization
    if dist_transform.max() > 0:
        dist_norm = (dist_transform / dist_transform.max() * 255).astype(np.uint8)
        # Apply colormap to distance (returns BGR)
        dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
        # Blend with tissue mask
        tissue_region = mask > 0
        vis[tissue_region] = dist_color[tissue_region]

    # Draw target contour in green (BGR: 0, 255, 0)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    # Colors for different tissues (BGR format)
    tissue_colors = [
        (0, 0, 255),    # Red - Tissue 1
        (255, 0, 0),    # Blue - Tissue 2
        (0, 255, 255),  # Yellow - Tissue 3
        (255, 0, 255),  # Magenta - Tissue 4
        (255, 255, 0),  # Cyan - Tissue 5
        (0, 165, 255),  # Orange - Tissue 6
        (147, 20, 255), # Pink - Tissue 7
        (0, 255, 0),    # Green - Tissue 8
    ]

    # Draw sample points colored by tissue ID with patch numbers
    for idx, point in enumerate(sample_points):
        if len(point) >= 3:
            x, y, tissue_id = point[0], point[1], int(point[2])
        else:
            x, y = point[0], point[1]
            tissue_id = 1
        color = tissue_colors[(tissue_id - 1) % len(tissue_colors)]
        cv2.circle(vis, (int(x), int(y)), 5, color, -1)
        # Add patch number
        cv2.putText(
            vis,
            str(idx),
            (int(x) + 7, int(y) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

    # Count tissues
    if sample_points and len(sample_points[0]) >= 3:
        num_tissues = len(set(int(p[2]) for p in sample_points))
    else:
        num_tissues = 1

    # Add text info
    if isinstance(distance_label, (int, float)):
        dist_text = f"{distance_label}um"
    else:
        dist_text = str(distance_label)
    cv2.putText(
        vis,
        f"Distance: {dist_text}, Tissues: {num_tissues}, Points: {len(sample_points)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # Save (vis is already BGR, cv2.imwrite expects BGR)
    cv2.imwrite(output_path, vis)


def generate_contour_patches(
    slide_path: str,
    output_dir: str,
    distance_from_edge_um: float | None = None,
    distance_fraction: float | None = None,
    arc_step_um: float = 500,
    patch_size: tuple[int, int] = (512, 512),
    level: int = 0,
    max_patches: int | None = None,
    select_tissues: list[int] | None = None,
    save_debug: bool = True,
    verbose: bool = True,
) -> int:
    """
    Generate patches along a contour at a specified distance from tissue edge.

    Supports two distance modes (exactly one must be specified):
    - Absolute: distance_from_edge_um - same distance for all tissues
    - Fractional: distance_fraction - distance as fraction of each tissue's diameter

    Args:
        slide_path: Path to whole slide image
        output_dir: Directory to save patches
        distance_from_edge_um: Target distance in microns (absolute mode)
        distance_fraction: Target distance as fraction of diameter, 0-0.5 (fractional mode)
        arc_step_um: Spacing between patches along contour in microns
        patch_size: Size of each patch in pixels (width, height)
        level: Slide level to extract from (0 = highest resolution)
        max_patches: Maximum number of patches to extract (None = no limit)
        select_tissues: List of tissue IDs to include (None = all tissues)
        save_debug: Save a visualization image
        verbose: Print progress info

    Returns:
        Number of patches generated
    """
    # Default to 1500μm if neither specified
    if distance_from_edge_um is None and distance_fraction is None:
        distance_from_edge_um = 1500.0
    # Open slide
    if verbose:
        print(f"Opening slide: {slide_path}")
    slide = openslide.OpenSlide(slide_path)

    # Get slide mpp
    try:
        slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    except (KeyError, ValueError):
        slide_mpp = 0.25
        if verbose:
            print(f"Warning: MPP not found, assuming {slide_mpp} μm/px")

    if verbose:
        print(f"Slide dimensions: {slide.dimensions}")
        print(f"Slide MPP: {slide_mpp} μm/px")
        print(f"Levels: {slide.level_count}")

    # Validate level
    if level < 0 or level >= slide.level_count:
        print(f"Error: Level {level} is invalid. Slide has {slide.level_count} levels (0-{slide.level_count - 1}).")
        slide.close()
        return 0

    # Generate tissue mask
    if verbose:
        print("\nGenerating tissue mask...")
    mask, mask_downsample, mask_mpp = get_tissue_mask(slide, verbose=verbose)

    # Get contour at target distance
    if verbose:
        if distance_from_edge_um is not None:
            print(f"\nFinding contour at {distance_from_edge_um}μm from edge...")
        else:
            print(f"\nFinding contour at {distance_fraction:.1%} of diameter from edge...")
    contours, dist_transform, mask_filled = get_contour_at_distance(
        mask,
        mpp=mask_mpp,
        distance_um=distance_from_edge_um,
        distance_fraction=distance_fraction,
        verbose=verbose,
    )

    if not contours:
        print("Error: No valid contours found. Tissue may be too small for target distance.")
        slide.close()
        return 0

    # Sort and label contours (assign tissue IDs based on top-to-bottom order)
    if verbose:
        print(f"\nLabeling tissues...")
    labeled_contours = sort_and_label_contours(contours, verbose=verbose)

    if not labeled_contours:
        print("Error: No valid contours to process.")
        slide.close()
        return 0

    # Filter to selected tissues BEFORE sampling (efficient)
    if select_tissues is not None:
        original_count = len(labeled_contours)
        labeled_contours = [(tid, c) for tid, c in labeled_contours if tid in select_tissues]
        if verbose:
            print(f"  Filtering to tissues {select_tissues}: {original_count} → {len(labeled_contours)} tissue(s)")
        if not labeled_contours:
            print(f"Error: No tissues remain after filtering to {select_tissues}")
            slide.close()
            return 0

    # Sample points along selected contours
    if verbose:
        print(f"\nSampling points with {arc_step_um}μm spacing...")
    arc_step_px = arc_step_um / mask_mpp

    # Validate arc_step to prevent infinite loop
    if arc_step_px <= 0:
        print(f"Error: arc_step must be positive (got {arc_step_um}μm = {arc_step_px}px)")
        slide.close()
        return 0

    sample_points = sample_contour_points(labeled_contours, arc_step_px, verbose=verbose)

    if not sample_points:
        print("Error: No sample points generated.")
        slide.close()
        return 0

    # Apply max_patches limit if specified
    if max_patches is not None and len(sample_points) > max_patches:
        if verbose:
            print(f"  Limiting from {len(sample_points)} to {max_patches} patches (evenly spaced)")
        # Select evenly spaced subset
        indices = np.linspace(0, len(sample_points) - 1, max_patches, dtype=int)
        sample_points = [sample_points[i] for i in indices]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save debug visualization
    if save_debug:
        debug_path = os.path.join(output_dir, "_debug_contour.png")
        # For display, use the absolute distance or indicate fractional
        display_distance = distance_from_edge_um if distance_from_edge_um is not None else f"{distance_fraction:.0%}"
        save_debug_visualization(
            mask_filled, dist_transform, contours, sample_points,
            debug_path, display_distance
        )
        if verbose:
            print(f"\nSaved debug visualization: {debug_path}")

    # Extract patches
    if verbose:
        print(f"\nExtracting {len(sample_points)} patches...")

    slide_name = Path(slide_path).stem
    patch_count = 0

    # Get level downsample factor for coordinate calculations
    level_downsample = slide.level_downsamples[level]

    # Patch size at level 0 coordinates (for centering and bounds)
    # read_region size is at target level, but location is always level 0
    level0_patch_width = int(patch_size[0] * level_downsample)
    level0_patch_height = int(patch_size[1] * level_downsample)

    # Calculate physical patch size in microns
    # At level N with downsample D, each pixel covers D * mpp microns
    patch_um_x = round(patch_size[0] * slide_mpp * level_downsample)
    patch_um_y = round(patch_size[1] * slide_mpp * level_downsample)

    if verbose:
        print(f"  Patch size: {patch_size[0]}x{patch_size[1]} px at level {level}")
        print(f"  Physical size: {patch_um_x}x{patch_um_y} μm")

    # Track per-tissue patch indices for filename numbering
    tissue_patch_indices = {}

    for i, point in enumerate(sample_points):
        # Unpack point (x, y, tissue_id)
        mask_x, mask_y = point[0], point[1]
        tissue_id = int(point[2]) if len(point) >= 3 else 1

        # Track patch index within this tissue
        if tissue_id not in tissue_patch_indices:
            tissue_patch_indices[tissue_id] = 0
        patch_idx = tissue_patch_indices[tissue_id]
        tissue_patch_indices[tissue_id] += 1

        # Convert mask coordinates to level 0 coordinates
        level0_center_x = int(mask_x * mask_downsample)
        level0_center_y = int(mask_y * mask_downsample)

        # Center the patch on this point (using level 0 dimensions)
        level0_x = level0_center_x - level0_patch_width // 2
        level0_y = level0_center_y - level0_patch_height // 2

        # Ensure within bounds (using level 0 dimensions)
        level0_x = max(0, min(level0_x, slide.dimensions[0] - level0_patch_width))
        level0_y = max(0, min(level0_y, slide.dimensions[1] - level0_patch_height))

        # Extract patch (location in level 0 coords, size in target level pixels)
        region = slide.read_region((level0_x, level0_y), level, patch_size)
        region_rgb = region.convert('RGB')

        # Generate filename with tissue ID and metadata
        if distance_from_edge_um is not None:
            distance_str = f"d{int(distance_from_edge_um)}um"
        else:
            distance_str = f"d{int(distance_fraction * 100)}pct"
        filename = (
            f"{slide_name}_"
            f"t{tissue_id:02d}_"
            f"{distance_str}_"
            f"p{patch_idx:04d}_"
            f"x{patch_um_x}um_y{patch_um_y}um.tiff"
        )
        out_path = os.path.join(output_dir, filename)

        region_rgb.save(out_path, 'TIFF')
        patch_count += 1

        if verbose and (i + 1) % 50 == 0:
            print(f"  Extracted {i + 1}/{len(sample_points)} patches...")

    if verbose:
        print(f"\nComplete! Generated {patch_count} patches")

    slide.close()
    return patch_count


def main():
    parser = argparse.ArgumentParser(
        description='Generate patches along tissue contour at specified distance from edge.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: 1500μm from edge, 500μm spacing, 512x512 patches
    python generate_patches_contour.py slide.svs -o out

    # Custom distance and spacing
    python generate_patches_contour.py slide.svs -o out --distance 1200 --arc-step 300

    # Larger patches
    python generate_patches_contour.py slide.svs -o out --size 1024 1024

    # Limit number of patches
    python generate_patches_contour.py slide.svs -o out --max-patches 100
        """
    )

    parser.add_argument(
        'slide_path',
        type=str,
        help='Path to the whole slide image file'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='out',
        help='Directory to save output patches (default: out)'
    )

    # Distance options (mutually exclusive)
    distance_group = parser.add_mutually_exclusive_group()
    distance_group.add_argument(
        '--distance',
        type=float,
        default=None,
        help='Distance from tissue edge in microns (default: 1500 if neither specified)'
    )
    distance_group.add_argument(
        '--distance-fraction',
        type=float,
        default=None,
        metavar='FRAC',
        help='Distance as fraction of tissue diameter, 0-0.5 (e.g., 0.25 for 1/4 from edge)'
    )

    parser.add_argument(
        '--arc-step',
        type=float,
        default=500,
        help='Spacing between patches along contour in microns (default: 500)'
    )

    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=('WIDTH', 'HEIGHT'),
        help='Patch size in pixels (default: 512 512)'
    )

    parser.add_argument(
        '--level',
        type=int,
        default=0,
        help='Slide level to extract from, 0=highest resolution (default: 0)'
    )

    parser.add_argument(
        '--max-patches',
        type=int,
        default=None,
        help='Maximum number of patches to generate (optional)'
    )

    parser.add_argument(
        '--select-tissues',
        type=str,
        default=None,
        help='Comma-separated list of tissue IDs to include (e.g., "1,3"). Default: all tissues.'
    )

    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Skip saving debug visualization image'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Validate slide path
    if not os.path.exists(args.slide_path):
        print(f"Error: Slide file not found: {args.slide_path}")
        sys.exit(1)

    # Validate parameters
    if args.distance is not None and args.distance <= 0:
        print(f"Error: --distance must be positive (got {args.distance})")
        sys.exit(1)

    if args.distance_fraction is not None:
        if args.distance_fraction <= 0 or args.distance_fraction > 0.5:
            print(f"Error: --distance-fraction must be between 0 and 0.5 (got {args.distance_fraction})")
            sys.exit(1)

    if args.arc_step <= 0:
        print(f"Error: --arc-step must be positive (got {args.arc_step})")
        sys.exit(1)

    if args.size[0] <= 0 or args.size[1] <= 0:
        print(f"Error: --size must be positive (got {args.size})")
        sys.exit(1)

    # Parse select_tissues
    select_tissues = None
    if args.select_tissues:
        try:
            select_tissues = [int(t.strip()) for t in args.select_tissues.split(',')]
        except ValueError:
            print(f"Error: --select-tissues must be comma-separated integers (got '{args.select_tissues}')")
            sys.exit(1)

    # Generate patches
    patch_count = generate_contour_patches(
        slide_path=args.slide_path,
        output_dir=args.output_dir,
        distance_from_edge_um=args.distance,
        distance_fraction=args.distance_fraction,
        arc_step_um=args.arc_step,
        patch_size=tuple(args.size),
        level=args.level,
        max_patches=args.max_patches,
        select_tissues=select_tissues,
        save_debug=not args.no_debug,
        verbose=not args.quiet,
    )

    sys.exit(0 if patch_count > 0 else 1)


if __name__ == '__main__':
    main()
