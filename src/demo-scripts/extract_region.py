#!/usr/bin/env python3
"""
Extract a region from a whole slide image at a specific zoom level.

This script demonstrates how to use OpenSlide to open and process Whole Slide Images,
extracting a specific region that can be used for machine learning applications.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import openslide
import tifffile
from PIL import Image


def extract_region(
    slide_path: str,
    level: int,
    width_percent: float,
    height_percent: float,
    output_size: tuple[int, int] = (512, 512),
    output_format: str = "tiff",
    quality: int = 100,
    output_dir: str = "out",
) -> None:
    """
    Extract a region from a whole slide image.

    Args:
        slide_path: Path to the whole slide image file
        level: Zoom level to extract from (0 is highest resolution)
        width_percent: Percentage of width for starting position (0-100)
        height_percent: Percentage of height for starting position (0-100)
        output_size: Size of the region to extract (width, height)
        output_format: Output format - 'tiff' (lossless tiled), 'png' (lossless), or 'jpeg' (lossy)
        quality: JPEG quality (1-100, only applies to jpeg format)
        output_dir: Directory to save the output image
    """
    # Open the slide
    print(f"Opening slide: {slide_path}")
    slide = openslide.OpenSlide(slide_path)

    # Print slide information
    print(f"\nSlide information:")
    print(f"  Dimensions: {slide.dimensions}")
    print(f"  Level count: {slide.level_count}")
    print(f"  Level dimensions: {slide.level_dimensions}")
    print(f"  Level downsamples: {slide.level_downsamples}")

    # Validate level
    if level < 0 or level >= slide.level_count:
        print(f"Error: Level {level} is out of range. Valid range: 0-{slide.level_count - 1}")
        sys.exit(1)

    # Validate percentages
    if not (0 <= width_percent <= 100) or not (0 <= height_percent <= 100):
        print("Error: Percentages must be between 0 and 100")
        sys.exit(1)

    # Get dimensions at level 0 (highest resolution)
    level0_width, level0_height = slide.dimensions

    # Calculate starting position at level 0 coordinates
    # OpenSlide always uses level 0 coordinates as reference
    start_x = int((width_percent / 100) * level0_width)
    start_y = int((height_percent / 100) * level0_height)

    # Ensure we don't go out of bounds
    max_x = level0_width - int(output_size[0] * slide.level_downsamples[level])
    max_y = level0_height - int(output_size[1] * slide.level_downsamples[level])
    start_x = min(start_x, max(0, max_x))
    start_y = min(start_y, max(0, max_y))

    print(f"\nExtracting region:")
    print(f"  Level: {level}")
    print(f"  Starting position (level 0 coords): ({start_x}, {start_y})")
    print(f"  Position as percentage: ({width_percent}%, {height_percent}%)")
    print(f"  Region size: {output_size}")
    print(f"  Downsample factor: {slide.level_downsamples[level]}")

    # Extract the region
    # read_region returns RGBA, location is in level 0 reference frame
    region = slide.read_region((start_x, start_y), level, output_size)

    # Convert RGBA to RGB
    region_rgb = region.convert('RGB')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine file extension based on format
    extension_map = {
        'tiff': '.tiff',
        'png': '.png',
        'jpeg': '.jpg'
    }
    extension = extension_map[output_format]

    # Generate output filename
    slide_name = Path(slide_path).stem
    output_filename = f"{slide_name}_L{level}_x{start_x}_y{start_y}_{output_size[0]}x{output_size[1]}{extension}"
    output_path = os.path.join(output_dir, output_filename)

    # Save based on format
    if output_format == 'tiff':
        # Convert to numpy array for tifffile
        region_array = np.array(region_rgb)

        # Save as tiled TIFF with compression
        # tile=(256, 256) creates a tiled TIFF for efficient access
        # compression='adobe_deflate' provides lossless compression
        # photometric='rgb' ensures proper color interpretation
        tifffile.imwrite(
            output_path,
            region_array,
            tile=(256, 256),
            compression='adobe_deflate',
            photometric='rgb',
            metadata={'axes': 'YXC'}
        )
        print(f"\nSaved extracted region to: {output_path} (TIFF, lossless, tiled)")

    elif output_format == 'png':
        # Save as PNG (lossless compression)
        region_rgb.save(output_path, 'PNG', optimize=True)
        print(f"\nSaved extracted region to: {output_path} (PNG, lossless)")

    elif output_format == 'jpeg':
        # Save as JPEG (lossy compression with quality setting)
        region_rgb.save(output_path, 'JPEG', quality=quality, optimize=True)
        print(f"\nSaved extracted region to: {output_path} (JPEG, quality={quality})")

    # Close the slide
    slide.close()


def main():
    parser = argparse.ArgumentParser(
        description='Extract a region from a whole slide image at a specific zoom level.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from center of slide at level 2 (default TIFF format)
  python extract_region.py slide.svs --level 2 --width-percent 50 --height-percent 50

  # Extract as PNG (lossless)
  python extract_region.py slide.svs --level 0 --width-percent 10 --height-percent 10 --format png

  # Extract as JPEG with quality 100 (lossy but smallest size)
  python extract_region.py slide.svs --level 0 --width-percent 10 --height-percent 10 --format jpeg --quality 100

  # Extract with custom size and output directory
  python extract_region.py slide.svs --level 1 --width-percent 25 --height-percent 75 --size 1024 1024 --output-dir results
        """
    )

    parser.add_argument(
        'slide',
        type=str,
        help='Path to the whole slide image file'
    )

    parser.add_argument(
        '--level',
        type=int,
        required=True,
        help='Zoom level to extract from (0 is highest resolution)'
    )

    parser.add_argument(
        '--width-percent',
        type=float,
        required=True,
        help='Percentage of width for starting position (0-100)'
    )

    parser.add_argument(
        '--height-percent',
        type=float,
        required=True,
        help='Percentage of height for starting position (0-100)'
    )

    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=('WIDTH', 'HEIGHT'),
        help='Size of the region to extract in pixels (default: 512 512)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['tiff', 'png', 'jpeg'],
        default='tiff',
        help='Output format: tiff (lossless, tiled), png (lossless), jpeg (lossy) (default: tiff)'
    )

    parser.add_argument(
        '--quality',
        type=int,
        default=100,
        metavar='N',
        help='JPEG quality (1-100, only applies to jpeg format) (default: 100)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='out',
        help='Directory to save the output image (default: out)'
    )

    args = parser.parse_args()

    # Check if slide file exists
    if not os.path.exists(args.slide):
        print(f"Error: Slide file not found: {args.slide}")
        sys.exit(1)

    # Validate quality range
    if not (1 <= args.quality <= 100):
        print("Error: Quality must be between 1 and 100")
        sys.exit(1)

    # Extract the region
    extract_region(
        slide_path=args.slide,
        level=args.level,
        width_percent=args.width_percent,
        height_percent=args.height_percent,
        output_size=tuple(args.size),
        output_format=args.format,
        quality=args.quality,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
