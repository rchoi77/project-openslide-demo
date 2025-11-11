import argparse
import os
import sys

from crop_tiff import extract_region

def main():
    parser = argparse.ArgumentParser(
        description='Extract many crops from a whole slide image.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """Usage examples:
        # Generate default size crops.
        python generate_patches.py slide.svs -x 10 90 -y 5 40 --step-size 10

        # Specify size and output directory.
        python generate_patches.py slide.svs -x 10 90 -y 5 40 --step-size 10 --size 3072 3072 -o outputs
        """
    )

    parser.add_argument(
        'slide_path',
        type=str,
        help='Path to the whole slide image file.'
    )

    parser.add_argument(
        'x', '--width-range',
        type=int,
        nargs=2,
        required=True,
        metavar=('START', 'END'),
        help='Percent values for where to start and stop width-wise.'
    )

    parser.add_argument(
        'y', '--height-range',
        type=int,
        nargs=2,
        required=True,
        metavar=('START', 'END'),
        help='Percent values for where to start and stop height-wise.'
    )

    parser.add_argument(
        '--step-size',
        type=int,
        required=True,
        help='Percent value for increment size between patch starting points.'
    )

    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        default=[3072, 3072],
        metavar=('WIDTH', 'HEIGHT'),
        help='Size of the region to extract in pixels (default: 3072 3072)'
    )

    parser.add_argument(
        'o', '--output-dir',
        type=str,
        default='out',
        help='Directory to save the output images (default: out)'
    )

    args = parser.parse_args()

    slidepath = args.slide_path
    if not os.path.exists(slidepath):
        print(f"Error: Slide file not found: {slidepath}")
        sys.exit(1)
    outdir = args.output_dir
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    start_x, end_x = args.width_range
    start_y, end_y = args.height_range
    step_size = args.step_size
    assert(start_x > 0 and end_x < 100 and start_y > 0 and end_y < 100 and step_size > 0 and step_size < 100)

    size = tuple(args.size)

    # Extract the regions
    for x in range(start_x, end_x, step_size):
        for y in range(start_y, end_y, step_size):
            extract_region(
                slide_path=slidepath,
                level=0,
                width_percent=x,
                height_percent=y,
                output_size=size,
                output_dir=outdir,
                verbose=False,
            )

if __name__ == '__main__':
    main()