# OpenSlide Demo

Command line tool using OpenSlide to extract regions from whole slide images. Supports TIFF (lossless, tiled), PNG (lossless), and JPEG (lossy) output formats.

### Setup:

### Requires: Python>=3.10

Install dependencies:
```bash
uv sync
# or
pip install -r requirements.txt
```

Free sample data [here](https://openslide.cs.cmu.edu/download/openslide-testdata/)

### Usage:

Level 0 corresponds to the highest resolution. Default: 512x512 pixels, TIFF format.

#### Extract 1024x1024 region as TIFF (default):
```bash
python src/demo-scripts/extract_region.py [slide.svs] --level 0 --width-percent 15 --height-percent 75 --size 1024 1024
```

#### Output formats:
```bash
--format tiff   # Lossless, tiled (default)
--format png    # Lossless
--format jpeg --quality 100  # Lossy, quality 1-100
```

---

Also comes with a provided web-viewer for whole slide images. 

`python src/deepzoom-server/deepzoom_server.py \[your_slide_image.svs\]`