#!/bin/bash
# first arg is input file, second is output directory

# ilastik pixel segmentation
/opt/ilastik/ilastik-1.4.0b21-gpu-Linux/run_ilastik.sh --headless --project=/opt/pytcher-plants/pytcherplants.ilp --output_format=tiff --output_filename_format=$2/{nickname}_segmented.tiff --export_source="Simple Segmentation" $1

# preprocess ilastik pixel segmentation results
python3 /opt/pytcher-plants/pytcher_plants/cli.py ilastik postpc -i $1 -m "$2/$(basename $1 | cut -d. -f1)_segmented.tiff" -o $2

# color analysis
# python3 /opt/pytcher-plants/pytcher_plants/cli.py colors analyze -i $2/preprocessed -o $2

# TODO ilastik growth point counting
# TODO ilastik pitcher counting
# TODO pitcher instance segmentation?
# TODO individual pitcher color analysis and trait measurement?
