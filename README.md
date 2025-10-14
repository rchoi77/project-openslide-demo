# OpenSlide Demo

Simple command line tool demo using OpenSlide to generate jpg images from whole slide image formats. Next step is to feed these images into preprocessing for the segmentation model.

### Setup:

### Requires: Python>=3.10

Install dependencies from requirements.txt

Free sample data [here](https://openslide.cs.cmu.edu/download/openslide-testdata/)

### Usage:

Level 0 corresponds to the highest resolution slide image (image slice will be most zoomed in). By default extracts 512x512 pixels.

#### Extract a 1024x1024 image with top-left corner starting near the bottom left of the original image at level 0 (most zoomed in):
`python main.py \[YOUR_SLIDE_IMAGE.svs\] --level 0 --width-percent 15 --height-percent 75 --size 1024 1024`

---

Also comes with a provided web-viewer for whole slide images. 

`python src/deepzoom-server/deepzoom_server.py \[your_slide_image.svs\]`