#!/usr/bin/env python
"""
Batch patch generation from local whole slide images.

Iterates through WSI files in a directory and generates patches using
contour-based sampling, saving results to per-slide folders.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from generate_patches_contour import generate_contour_patches

# Supported WSI extensions
WSI_EXTENSIONS = {".tiff", ".tif", ".ndpi", ".svs", ".mrxs", ".scn", ".vsi"}


def find_wsi_files(input_dir: Path, extensions: set[str]) -> list[Path]:
    """Find all WSI files in the input directory."""
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*{ext}"))
        files.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Batch patch generation from local whole slide images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults (1/3 radius from edge)
  %(prog)s /path/to/wsi/folder -o /path/to/output

  # Custom absolute distance
  %(prog)s /path/to/wsi/folder -o ./out --distance 1200

  # Custom fractional distance
  %(prog)s /path/to/wsi/folder -o ./out --distance-fraction 0.25

  # Limit patches per slide
  %(prog)s /path/to/wsi/folder -o ./out --max-patches 50
        """,
    )

    # Required arguments
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing WSI files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("out"),
        help="Output directory for results (default: out)",
    )

    # Patch generation parameters
    patch_group = parser.add_argument_group("Patch Generation")
    distance_mode = patch_group.add_mutually_exclusive_group()
    distance_mode.add_argument(
        "--distance",
        type=float,
        help="Distance from edge in micrometers (overrides default fraction)",
    )
    distance_mode.add_argument(
        "--distance-fraction",
        type=float,
        dest="distance_fraction",
        default=0.167,
        help="Distance as fraction of tissue diameter (default: 0.167 = 1/3 radius)",
    )
    patch_group.add_argument(
        "--arc-step",
        type=float,
        default=500,
        dest="arc_step",
        help="Spacing between samples in micrometers (default: 500)",
    )
    patch_group.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        metavar=("WIDTH", "HEIGHT"),
        help="Patch dimensions in pixels (default: 1024 1024)",
    )
    patch_group.add_argument(
        "--level",
        type=int,
        default=0,
        help="Slide pyramid level to extract from (default: 0 = highest resolution)",
    )
    patch_group.add_argument(
        "--select-tissues",
        type=int,
        nargs="+",
        dest="select_tissues",
        help="Process only specific tissue IDs (e.g., --select-tissues 1 2)",
    )
    patch_group.add_argument(
        "--max-patches",
        type=int,
        dest="max_patches",
        help="Maximum patches per slide (default: no limit)",
    )
    patch_group.add_argument(
        "--max-patches-mode",
        action="store_true",
        dest="max_patches_mode",
        help="Auto-calculate spacing for max patches without overlap",
    )

    # Other options
    other_group = parser.add_argument_group("Other Options")
    other_group.add_argument(
        "--extensions",
        type=str,
        default=".tiff,.ndpi,.svs",
        help="Comma-separated file extensions to process (default: .tiff,.ndpi,.svs)",
    )
    other_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"Error: Not a directory: {args.input_dir}")
        sys.exit(1)

    # Parse extensions
    extensions = {ext.strip().lower() for ext in args.extensions.split(",")}
    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}

    # Find WSI files
    wsi_files = find_wsi_files(args.input_dir, extensions)

    if not wsi_files:
        print(f"Error: No WSI files found in {args.input_dir}")
        print(f"Looking for extensions: {', '.join(sorted(extensions))}")
        sys.exit(1)

    # Determine distance mode
    if args.distance is not None:
        distance_um = args.distance
        distance_fraction = None
    else:
        distance_um = None
        distance_fraction = args.distance_fraction

    # Print configuration
    if verbose:
        print(f"Input directory: {args.input_dir}")
        print(f"Found {len(wsi_files)} WSI files")
        print(f"Output directory: {args.output}")
        if distance_fraction:
            print(f"Distance mode: {distance_fraction:.1%} of tissue diameter (1/3 radius)")
        else:
            print(f"Distance mode: {distance_um} um from edge")
        print(f"Patch size: {args.size[0]}x{args.size[1]} px")
        print(f"Arc step: {args.arc_step} um")
        print("=" * 60)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process each file
    success_count = 0
    fail_count = 0
    total_patches = 0

    for idx, wsi_path in enumerate(wsi_files, 1):
        print(f"\n[{idx}/{len(wsi_files)}] {wsi_path.name}")

        # Create per-slide output directory
        slide_stem = wsi_path.stem
        slide_output_dir = args.output / slide_stem

        # Generate patches
        try:
            patch_count = generate_contour_patches(
                slide_path=str(wsi_path),
                output_dir=str(slide_output_dir),
                distance_from_edge_um=distance_um,
                distance_fraction=distance_fraction,
                arc_step_um=args.arc_step,
                patch_size=tuple(args.size),
                level=args.level,
                select_tissues=args.select_tissues,
                max_patches=args.max_patches,
                max_patches_mode=args.max_patches_mode,
                verbose=verbose,
            )
            success_count += 1
            total_patches += patch_count
            print(f"  Generated {patch_count} patches")

        except Exception as e:
            print(f"  Error processing slide: {e}")
            fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed: {success_count}/{len(wsi_files)} slides")
    print(f"Failed: {fail_count}/{len(wsi_files)} slides")
    print(f"Total patches: {total_patches}")
    print(f"Output: {args.output}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
