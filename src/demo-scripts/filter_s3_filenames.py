"""
Filter S3 filenames: keep only _Te (testis), exclude unwanted species/types.
"""

import re

INPUT_FILE = "slides_to_process.txt"
OUTPUT_FILE = "slides_filtered.txt"

# Must contain this pattern
INCLUDE_PATTERN = r"_Te"

# Exclude these even if they match include pattern
EXCLUDE_PATTERNS = [
    r"ovary",
    r"overy",
    r"uterus",
    r"_rat",
    r"_cat",
    r"ihc",
    r"rex",
]


def should_include(filename: str) -> bool:
    """Check if filename should be included."""
    lower = filename.lower()

    # Must match include pattern
    if not re.search(INCLUDE_PATTERN, filename, re.IGNORECASE):
        return False

    # Must not match any exclude pattern
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, lower):
            return False

    return True


def main():
    with open(INPUT_FILE) as f:
        filenames = [line.strip() for line in f if line.strip()]

    original_count = len(filenames)
    filtered = [fn for fn in filenames if should_include(fn)]

    with open(OUTPUT_FILE, "w") as f:
        for fn in filtered:
            f.write(fn + "\n")

    print(f"Original: {original_count}")
    print(f"Kept: {len(filtered)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
