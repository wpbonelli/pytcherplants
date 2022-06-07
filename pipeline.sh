#!/bin/bash

# first arg is input file, second is output directory
input=$1
output=$2

# ilastik pixel segmentation
pypl pixel_classification classify -i "$input" -o "$output"
# /opt/ilastik/ilastik-1.4.0b21-gpu-Linux/run_ilastik.sh --headless --project="/opt/pytcherplants/pytcherplants.ilp" --output_format="tiff" --output_filename_format="$output/{nickname}.segmented.tiff" --export_source="Simple Segmentation" "$input"

# postprocess ilastik pixel segmentation results
base="$output/$(basename "$input" | rev | cut -d. -f2- | rev)"
segmented="$base.segmented.tiff"
echo "$segmented"
pypl pixel_classification postprocess -i "$input" -m "$segmented" -o "$output"
input="$base.masked.jpg"

# geometric trait and color analysis from masked plant tissues
pypl colors analyze -i "$input" -o "$output"

# TODO ilastik growth point counting
# TODO ilastik pitcher counting
# TODO pitcher instance segmentation?
# TODO individual pitcher color analysis and trait measurement?
