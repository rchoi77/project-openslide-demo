  # Demo Scripts:

  ## extract_region.py
  Level 0 is highest resolution slide image (image slice will be most zoomed in).
  
  ### Usage Examples:
  #### Extract from center at level 2
  `python extract_region.py slide.svs --level 2 --width-percent 50 --height-percent 50`

  #### Extract from top-left at highest resolution with custom 1024x1024 size
  `python extract_region.py slide.svs --level 0 --width-percent 10 --height-percent 10
  --size 1024 1024`

  #### Custom output directory
  `python extract_region.py slide.svs --level 1 --width-percent 25 --height-percent 75
  --output-dir results`

