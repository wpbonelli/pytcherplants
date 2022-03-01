#!/bin/bash
# first arg is input file, second is output directory

# ilastik pixel segmentation
/opt/ilastik/ilastik-1.4.0b21-gpu-Linux/run_ilastik.sh --headless --project=/opt/pytcher-plants/pitcherplants.ilp --output_format=tiff --output_filename_format=$2/{nickname}_segmented.tiff --export_source="Simple Segmentation" $1

# preprocessing
# mkdir -p $OUTPUT/preprocessed
# python3 /opt/pytcher-plants/pytcher_plants/cli.py preprocess -i $1 -o $2/preprocessed

# color analysis
# python3 /opt/pytcher-plants/pytcher_plants/cli.py colors analyze -i $2/preprocessed -o $2

# TODO ilastik growth point counting
# TODO ilastik pitcher counting
# TODO pitcher instance segmentation?
# TODO individual pitcher color analysis and trait measurement?
