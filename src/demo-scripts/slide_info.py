import argparse
import sys

import openslide


def print_slide_info(slide_path: str) -> None:
    """
    Print information about a whole slide image.

    Args:
        slide_path: Path to the whole slide image file
    """
    # Open the slide
    print(f"Opening slide: {slide_path}")
    slide = openslide.OpenSlide(slide_path)

    try:
        mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
    except:
        mpp_x, mpp_y = None, None

    # Print slide information
    print(f"\nSlide information:")
    print(f"  Dimensions: {slide.dimensions}")
    print(f"  Level count: {slide.level_count}")
    print(f"  Level dimensions: {slide.level_dimensions}")
    print(f"  Level downsamples: {slide.level_downsamples}")
    if mpp_x and mpp_y:
        print(f"  Microns per pixel: x-{mpp_x}, y-{mpp_y}")

    # Close the slide
    slide.close()


def main():
    parser = argparse.ArgumentParser(
        description='Print information about a whole slide image.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """Examples:
        # Print slide information
        python slide_info.py slide.svs
        """
    )

    parser.add_argument(
        'slide_path',
        type=str,
        help='Path to the whole slide image file.'
    )

    args = parser.parse_args()

    # Check if slide file exists
    import os
    if not os.path.exists(args.slide_path):
        print(f"Error: Slide file not found: {args.slide_path}")
        sys.exit(1)

    # Print slide information
    print_slide_info(slide_path=args.slide_path)


if __name__ == '__main__':
    main()
