"""
Stitch Morphle tiles at full resolution using scan_meta_v2.json offsets.

Usage:
    python morphle_stitch.py /path/to/morphle_folder --center 20,80 --grid 5
    python morphle_stitch.py /path/to/morphle_folder --center 20,80 --grid 5 --output patch.png
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from collections import deque


def load_scan_meta(folder: Path) -> dict:
    """Load stitching metadata."""
    meta_path = folder / "res" / "scan_meta_v2.json"
    with open(meta_path) as f:
        return json.load(f)


def build_tile_positions(scan_meta: dict, seed_col: int, seed_row: int) -> dict:
    """
    Build global tile positions using BFS from seed tile.
    Returns dict mapping (col, row) -> (x, y) pixel position.
    """
    positions = {(seed_col, seed_row): (0, 0)}
    queue = deque([(seed_col, seed_row)])
    visited = {(seed_col, seed_row)}

    while queue:
        col, row = queue.popleft()
        key = f"x{col}y{row}"

        if key not in scan_meta:
            continue

        entry = scan_meta[key]
        curr_x, curr_y = positions[(col, row)]

        # Right neighbor
        right_key = f"x{col+1}y{row}"
        if right_key in scan_meta and (col+1, row) not in visited:
            right_entry = scan_meta[right_key]
            if right_entry.get("left", {}).get("success"):
                tx = right_entry["left"]["tx"]
                ty = right_entry["left"]["ty"]
                positions[(col+1, row)] = (curr_x + tx, curr_y + ty)
                visited.add((col+1, row))
                queue.append((col+1, row))

        # Left neighbor
        if entry.get("left", {}).get("success") and (col-1, row) not in visited:
            tx = entry["left"]["tx"]
            ty = entry["left"]["ty"]
            positions[(col-1, row)] = (curr_x - tx, curr_y - ty)
            visited.add((col-1, row))
            queue.append((col-1, row))

        # Bottom neighbor
        bottom_key = f"x{col}y{row+1}"
        if bottom_key in scan_meta and (col, row+1) not in visited:
            bottom_entry = scan_meta[bottom_key]
            if bottom_entry.get("top", {}).get("success"):
                tx = bottom_entry["top"]["tx"]
                ty = bottom_entry["top"]["ty"]
                positions[(col, row+1)] = (curr_x + tx, curr_y + ty)
                visited.add((col, row+1))
                queue.append((col, row+1))

        # Top neighbor
        if entry.get("top", {}).get("success") and (col, row-1) not in visited:
            tx = entry["top"]["tx"]
            ty = entry["top"]["ty"]
            positions[(col, row-1)] = (curr_x - tx, curr_y - ty)
            visited.add((col, row-1))
            queue.append((col, row-1))

    return positions


def stitch_tiles(
    folder: Path,
    tile_positions: dict,
    center_col: int,
    center_row: int,
    grid_size: int = 5,
) -> np.ndarray:
    """
    Stitch a grid of tiles centered on (center_col, center_row).
    Returns the stitched image as numpy array.
    """
    tile_dir = folder / "tiled" / "8"
    raw_w, raw_h = 2976, 1968
    radius = grid_size // 2

    # Select tiles in grid
    grid_tiles = {}
    for dc in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            tc, tr = center_col + dc, center_row + dr
            if (tc, tr) in tile_positions:
                grid_tiles[(tc, tr)] = tile_positions[(tc, tr)]

    if not grid_tiles:
        raise ValueError(f"No tiles found around ({center_col}, {center_row})")

    # Calculate canvas bounds
    positions = list(grid_tiles.values())
    min_x = min(p[0] for p in positions)
    max_x = max(p[0] + raw_w for p in positions)
    min_y = min(p[1] for p in positions)
    max_y = max(p[1] + raw_h for p in positions)

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    print(f"Stitching {len(grid_tiles)} tiles -> {canvas_w:,} × {canvas_h:,} px")

    # Create canvas
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Place tiles
    for (tc, tr), (px, py) in grid_tiles.items():
        tile_path = tile_dir / f"x{tc}y{tr}.jpg"
        if tile_path.exists():
            tile = np.array(Image.open(tile_path))
            x = px - min_x
            y = py - min_y
            h = min(raw_h, canvas_h - y)
            w = min(raw_w, canvas_w - x)
            if h > 0 and w > 0 and y >= 0 and x >= 0:
                canvas[y:y+h, x:x+w] = tile[:h, :w]

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Stitch Morphle tiles at full resolution")
    parser.add_argument("folder", type=Path, help="Path to Morphle slide folder")
    parser.add_argument("--center", type=str, required=True, help="Center tile as col,row (e.g., 20,80)")
    parser.add_argument("--grid", type=int, default=5, help="Grid size (default: 5 for 5x5)")
    parser.add_argument("--output", type=Path, default=Path("stitched.png"), help="Output file (default: stitched.png)")
    args = parser.parse_args()

    # Parse center
    center_col, center_row = map(int, args.center.split(","))

    # Load metadata
    print(f"Loading metadata from {args.folder}")
    scan_meta = load_scan_meta(args.folder)

    # Build positions
    print(f"Building tile position map...")
    tile_positions = build_tile_positions(scan_meta, center_col, center_row)
    print(f"Computed positions for {len(tile_positions)} tiles")

    # Stitch
    stitched = stitch_tiles(
        args.folder,
        tile_positions,
        center_col,
        center_row,
        args.grid,
    )

    # Save
    Image.fromarray(stitched).save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
