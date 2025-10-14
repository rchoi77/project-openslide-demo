import sys
from pathlib import Path

# Add the src directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent / "src" / "demo-scripts"))

from extract_region import main as extract_region_main


def main():
    """Entry point that delegates to extract_region script."""
    extract_region_main()


if __name__ == "__main__":
    main()
