import argparse
import os
import re
import sys
from pathlib import Path

from PIL import Image


def parse_microns_from_filename(filename: str) -> tuple[int, int] | None:
    """
    Extract micron dimensions from filename pattern like 'x569um_y569um'.
    Returns (width_um, height_um) or None if not found.
    """
    match = re.search(r'x(\d+)um_y(\d+)um', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def generate_patches(
    image_path: str,
    output_dir: str,
    patch_size: tuple[int, int] = (512, 512),
    stride_percent: int = 10,
    verbose: bool = True,
) -> int:
    """
    Generate a grid of patches from an image.

    Args:
        image_path: Path to the source image
        output_dir: Directory to save patches
        patch_size: Size of each patch (width, height) in pixels
        stride_percent: Stride as percentage of image dimensions
        verbose: Print progress info

    Returns:
        Number of patches generated
    """
    # Load image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Get source image name (without extension)
    source_name = Path(image_path).stem

    # Parse micron info from filename
    microns = parse_microns_from_filename(source_name)
    if microns:
        source_um_x, source_um_y = microns
        # Calculate microns per pixel
        mpp_x = source_um_x / img_width
        mpp_y = source_um_y / img_height
        # Calculate patch size in microns
        patch_um_x = round(patch_size[0] * mpp_x)
        patch_um_y = round(patch_size[1] * mpp_y)
    else:
        # Fallback if no micron info in filename
        patch_um_x = patch_um_y = 0

    if verbose:
        print(f"Processing: {source_name}")
        print(f"  Image size: {img_width}x{img_height}")
        if microns:
            print(f"  Source microns: {source_um_x}um x {source_um_y}um")
            print(f"  Patch microns: {patch_um_x}um x {patch_um_y}um")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate patches
    patch_count = 0

    # Calculate positions (percentage-based)
    x_positions = list(range(0, 100 - int(patch_size[0] / img_width * 100) + 1, stride_percent))
    y_positions = list(range(0, 100 - int(patch_size[1] / img_height * 100) + 1, stride_percent))

    for x_percent in x_positions:
        for y_percent in y_positions:
            # Calculate pixel position
            x_px = int(x_percent / 100 * img_width)
            y_px = int(y_percent / 100 * img_height)

            # Ensure we don't exceed image bounds
            if x_px + patch_size[0] > img_width:
                x_px = img_width - patch_size[0]
            if y_px + patch_size[1] > img_height:
                y_px = img_height - patch_size[1]

            # Extract patch
            patch = img.crop((x_px, y_px, x_px + patch_size[0], y_px + patch_size[1]))

            # Generate filename
            if patch_um_x and patch_um_y:
                filename = f"{source_name}_patch_x{x_percent}_y{y_percent}_x{patch_um_x}um_y{patch_um_y}um.tiff"
            else:
                filename = f"{source_name}_patch_x{x_percent}_y{y_percent}.tiff"

            output_path = os.path.join(output_dir, filename)
            patch.save(output_path, 'TIFF')
            patch_count += 1

    if verbose:
        print(f"  Generated {patch_count} patches")

    img.close()
    return patch_count


def main():
    parser = argparse.ArgumentParser(
        description='Generate patches from all images in a directory.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all images with default settings (512x512, 10% stride)
    python batch_generate_patches.py /path/to/images -o output

    # Custom patch size and stride
    python batch_generate_patches.py /path/to/images -o output --size 256 256 --stride 20
        """
    )

    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing source images'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='output',
        help='Base output directory (default: output)'
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
        '--stride',
        type=int,
        default=10,
        help='Stride as percentage of image size (default: 10)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=True,
        help='Print progress info'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Find all TIFF files
    tiff_files = list(Path(args.input_dir).glob('*.tiff')) + list(Path(args.input_dir).glob('*.tif'))

    if not tiff_files:
        print(f"Error: No TIFF files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(tiff_files)} TIFF files")
    print(f"Patch size: {args.size[0]}x{args.size[1]}")
    print(f"Stride: {args.stride}%")
    print()

    total_patches = 0

    for tiff_path in sorted(tiff_files):
        # Create output subdirectory named after the source image
        image_name = tiff_path.stem
        image_output_dir = os.path.join(args.output_dir, image_name)

        patches = generate_patches(
            image_path=str(tiff_path),
            output_dir=image_output_dir,
            patch_size=tuple(args.size),
            stride_percent=args.stride,
            verbose=args.verbose,
        )
        total_patches += patches

    print()
    print(f"Complete! Generated {total_patches} patches from {len(tiff_files)} images")


if __name__ == '__main__':
    main()
