#!/bin/bash

# first arg is input file, second is output directory
input=$1
output=$2
base="$output/$(basename "$input" | rev | cut -d. -f2- | rev)"

pypl pixel_classification classify -i "$input" -o "$output"
pypl pixel_classification postprocess -i "$input" -m "$base.segmented.tiff" -o "$output"
pypl colors analyze -i "$base.masked.jpg" -o "$output"

# TODO ilastik growth point counting
# TODO ilastik pitcher counting
# TODO pitcher instance segmentation?
# TODO individual pitcher color analysis and trait measurement?
