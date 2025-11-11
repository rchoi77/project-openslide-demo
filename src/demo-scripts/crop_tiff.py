import argparse
import os
import sys
from pathlib import Path

import openslide
from PIL import Image


def extract_region(
    slide_path: str,
    level: int,
    width_percent: float,
    height_percent: float,
    output_size: tuple[int, int] = (3, 3),
    output_dir: str = "out",
    verbose: bool = True,
) -> None:
    """
    Extract a region from a whole slide image.

    Args:
        slide_path: Path to the whole slide image file
        level: Zoom level to extract from (0 is highest resolution)
        width_percent: Percentage of width for starting position (0-100)
        height_percent: Percentage of height for starting position (0-100)
        output_size: Size of the region to extract (width, height)
        output_dir: Directory to save the output image
    """
    # Open the slide
    if verbose: print(f"Opening slide: {slide_path}")
    slide = openslide.OpenSlide(slide_path)

    try:
        mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
    except:
        mpp_x, mpp_y = None, None

    if verbose:
        # Print slide information
        print(f"\nSlide information:")
        print(f"  Dimensions: {slide.dimensions}")
        print(f"  Level count: {slide.level_count}")
        print(f"  Level dimensions: {slide.level_dimensions}")
        print(f"  Level downsamples: {slide.level_downsamples}")
        if mpp_x and mpp_y:
            print(f"Microns per pixel: x-{mpp_x}, y-{mpp_y}")

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

    if mpp_x and mpp_y:
        x_microns = round(output_size[0] * mpp_x)
        y_microns = round(output_size[1] * mpp_y)

    if verbose:
        print(f"\nExtracting region:")
        print(f"  Level: {level}")
        print(f"  Starting position (level 0 coords): ({start_x}, {start_y})")
        print(f"  Position as percentage: ({width_percent}%, {height_percent}%)")
        print(f"  Region size: {output_size[0]}x{output_size[1]}")
        print(f"  Downsample factor: {slide.level_downsamples[level]}")
        if x_microns and y_microns:
            print(f"Real dimensions: {x_microns}um x {y_microns}um")

    # Extract the region
    # read_region returns RGBA, location is in level 0 reference frame
    region = slide.read_region((start_x, start_y), level, output_size)

    # Convert RGBA to RGB
    region_rgb = region.convert('RGB')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    slide_name = Path(slide_path).stem
    output_filename = f"{slide_name}_x{width_percent}%_y{height_percent}%_x{x_microns}um_y{y_microns}um.tiff"
    output_path = os.path.join(output_dir, output_filename)

    # Save as JPEG
    region_rgb.save(output_path, 'TIFF', quality=100)
    if verbose: print(f"\nSaved extracted region to: {output_path}")

    # Close the slide
    slide.close()


def main():
    parser = argparse.ArgumentParser(
        description='Extract a region from a whole slide image.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """Examples:
        # Extract from center of slide at zoom level 2.
        python crop_tiff.py slide.svs -x 50 -y 50 --level 2

        # Extract area near top left at highest resolution with custom size.
        python crop_tiff.py slide.svs -x 10 -y 10 --size 1024 1024

        # Extract to custom output directory with verbose output.
        python crop_tiff.py slide.svs -x 25 -y 75 -o results -v
        """
    )

    parser.add_argument(
        'slide_path',
        type=str,
        help='Path to the whole slide image file.'
    )

    parser.add_argument(
        '-x', '--width-percent',
        type=float,
        required=True,
        help='Percentage of width for starting position (0-100).'
    )

    parser.add_argument(
        '-y', '--height-percent',
        type=float,
        required=True,
        help='Percentage of height for starting position (0-100).'
    )

    parser.add_argument(
        '--level',
        type=int,
        default=0,
        help='Zoom level to extract from (0 is highest resolution).'
    )

    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        default=[4096, 4096],
        metavar=('WIDTH', 'HEIGHT'),
        help='Size of the region to extract in pixels (default: 512 512).'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='out',
        help='Directory to save the output image (default: out).'
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", default=True, help="Allow output or no."
    )

    args = parser.parse_args()

    # Check if slide file exists
    if not os.path.exists(args.slide_path):
        print(f"Error: Slide file not found: {args.slide_path}")
        sys.exit(1)

    # Extract the region
    extract_region(
        slide_path=args.slide_path,
        level=args.level,
        width_percent=args.width_percent,
        height_percent=args.height_percent,
        output_size=tuple(args.size),
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
