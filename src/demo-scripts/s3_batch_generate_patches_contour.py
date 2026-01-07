#!/usr/bin/env python
"""
Batch patch generation from S3-hosted whole slide images.

Downloads TIFF files from AWS S3, generates patches using contour-based
sampling, saves results to per-slide folders, and cleans up downloaded files.

Uses AWS credentials from the default boto3 credential chain:
- Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- ~/.aws/credentials file
- IAM role (if running on AWS)
"""

import argparse
import sys
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from generate_patches_contour import generate_contour_patches


class DownloadProgressCallback:
    """Callback to display download progress."""

    def __init__(self, filename: str, total_size: int | None = None):
        self.filename = filename
        self.total_size = total_size
        self.downloaded = 0

    def __call__(self, bytes_amount: int):
        self.downloaded += bytes_amount
        if self.total_size:
            percent = (self.downloaded / self.total_size) * 100
            print(f"\r  Downloading {self.filename}: {percent:.1f}%", end="", flush=True)
        else:
            mb = self.downloaded / (1024 * 1024)
            print(f"\r  Downloading {self.filename}: {mb:.1f} MB", end="", flush=True)


def download_from_s3(
    s3_client,
    bucket: str,
    key: str,
    local_path: str,
    verbose: bool = True,
) -> bool:
    """
    Download a file from S3.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key
        local_path: Local path to save file
        verbose: Print progress

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get file size for progress
        head = s3_client.head_object(Bucket=bucket, Key=key)
        total_size = head.get("ContentLength")

        if verbose:
            size_mb = total_size / (1024 * 1024) if total_size else 0
            print(f"  File size: {size_mb:.1f} MB")

        # Download with progress callback
        callback = DownloadProgressCallback(Path(key).name, total_size) if verbose else None
        s3_client.download_file(bucket, key, local_path, Callback=callback)

        if verbose:
            print()  # Newline after progress

        return True

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "404":
            print(f"  Error: File not found in S3: s3://{bucket}/{key}")
        else:
            print(f"  Error downloading from S3: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch patch generation from S3-hosted whole slide images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s slides.txt -o ./out

  # With custom patch parameters
  %(prog)s slides.txt -o ./out --distance 1500 --size 1024 1024

  # Use fractional distance (25%% of tissue diameter)
  %(prog)s slides.txt -o ./out --distance-fraction 0.25

  # Keep downloaded files for inspection
  %(prog)s slides.txt -o ./out --keep-downloads --download-dir ./downloads
        """,
    )

    # Required arguments
    parser.add_argument(
        "filelist",
        help="Text file with one TIFF filename per line",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for results (creates per-slide subdirectories)",
    )

    # S3 configuration
    s3_group = parser.add_argument_group("S3 Configuration")
    s3_group.add_argument(
        "--bucket",
        default="morphle-epivara-wsi",
        help="S3 bucket name (default: morphle-epivara-wsi)",
    )
    s3_group.add_argument(
        "--prefix",
        default="morphle-tiff/",
        help="S3 key prefix (default: morphle-tiff/)",
    )

    # Patch generation parameters
    patch_group = parser.add_argument_group("Patch Generation")
    distance_mode = patch_group.add_mutually_exclusive_group()
    distance_mode.add_argument(
        "--distance",
        type=float,
        default=1500,
        help="Distance from edge in micrometers (default: 1500)",
    )
    distance_mode.add_argument(
        "--distance-fraction",
        type=float,
        dest="distance_fraction",
        help="Distance as fraction of tissue diameter (0-0.5), alternative to --distance",
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

    # Other options
    other_group = parser.add_argument_group("Other Options")
    other_group.add_argument(
        "--keep-downloads",
        action="store_true",
        dest="keep_downloads",
        help="Don't delete downloaded files after processing",
    )
    other_group.add_argument(
        "--download-dir",
        dest="download_dir",
        help="Directory for downloads (default: temporary directory)",
    )
    other_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Read filenames from text file
    filelist_path = Path(args.filelist)
    if not filelist_path.exists():
        print(f"Error: File list not found: {filelist_path}")
        sys.exit(1)

    filenames = []
    with open(filelist_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if any(item in line.lower() for item in ["rat", "cat", "cattle", "overy"]):
                    print(f"Skipping {line} because it contains filter words")
                    continue
                filenames.append(line)

    if not filenames:
        print(f"Error: No filenames found in {filelist_path}")
        sys.exit(1)

    if verbose:
        print(f"Found {len(filenames)} files to process")
        print(f"S3 source: s3://{args.bucket}/{args.prefix}")
        print(f"Output directory: {args.output}")
        print("-" * 60)

    # Initialize S3 client
    try:
        s3_client = boto3.client("s3")
        # Test credentials by listing bucket (lightweight operation)
        s3_client.head_bucket(Bucket=args.bucket)
    except NoCredentialsError:
        print("Error: AWS credentials not found.")
        print("Configure credentials via:")
        print("  - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        print("  - AWS credentials file: ~/.aws/credentials")
        print("  - IAM role (if running on AWS)")
        sys.exit(1)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "403":
            print(f"Error: Access denied to bucket '{args.bucket}'")
        elif error_code == "404":
            print(f"Error: Bucket '{args.bucket}' not found")
        else:
            print(f"Error accessing S3: {e}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup download directory
    if args.download_dir:
        download_dir = Path(args.download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        temp_dir_context = None
    else:
        temp_dir_context = tempfile.TemporaryDirectory()
        download_dir = Path(temp_dir_context.name)

    # Determine distance mode
    if args.distance_fraction is not None:
        distance_um = None
        distance_fraction = args.distance_fraction
        if verbose:
            print(f"Distance mode: {distance_fraction:.0%} of tissue diameter")
    else:
        distance_um = args.distance
        distance_fraction = None
        if verbose:
            print(f"Distance mode: {distance_um} μm from edge")

    if verbose:
        print(f"Patch size: {args.size[0]}x{args.size[1]} px")
        print(f"Arc step: {args.arc_step} μm")
        print("=" * 60)

    # Process each file
    success_count = 0
    fail_count = 0
    total_patches = 0

    try:
        for idx, filename in enumerate(filenames, 1):
            print(f"\n[{idx}/{len(filenames)}] {filename}")

            # Construct S3 key
            s3_key = f"{args.prefix}{filename}"
            local_path = download_dir / filename

            # Download from S3
            if not download_from_s3(s3_client, args.bucket, s3_key, str(local_path), verbose):
                fail_count += 1
                continue

            # Create per-slide output directory
            slide_stem = Path(filename).stem
            slide_output_dir = output_dir / slide_stem

            # Generate patches
            try:
                patch_count = generate_contour_patches(
                    slide_path=str(local_path),
                    output_dir=str(slide_output_dir),
                    distance_from_edge_um=distance_um,
                    distance_fraction=distance_fraction,
                    arc_step_um=args.arc_step,
                    patch_size=tuple(args.size),
                    level=args.level,
                    select_tissues=args.select_tissues,
                    max_patches=args.max_patches,
                    verbose=verbose,
                )
                success_count += 1
                total_patches += patch_count
                print(f"  Generated {patch_count} patches")

            except Exception as e:
                print(f"  Error processing slide: {e}")
                fail_count += 1

            finally:
                # Cleanup downloaded file (unless --keep-downloads)
                if not args.keep_downloads and local_path.exists():
                    local_path.unlink()
                    if verbose:
                        print(f"  Cleaned up: {filename}")

    finally:
        # Cleanup temp directory if used
        if temp_dir_context:
            temp_dir_context.cleanup()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed: {success_count}/{len(filenames)} slides")
    print(f"Failed: {fail_count}/{len(filenames)} slides")
    print(f"Total patches: {total_patches}")
    print(f"Output: {output_dir}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
