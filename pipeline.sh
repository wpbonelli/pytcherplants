#!/bin/bash

# whether to use ilastik for pixel segmentation
ilastik=0
while getopts 'i' opt; do
    case $opt in
        i) ilastik=1 ;;
        *) echo 'Error in command line parsing' >&2
           exit 1
    esac
done
shift "$(( OPTIND - 1 ))"

# first arg is input file, second is output directory
input=$1
output=$2

if [[ "$ilastik" -eq 1 ]]; then
  # ilastik pixel segmentation
  /opt/ilastik/ilastik-1.4.0b21-gpu-Linux/run_ilastik.sh --headless --project="/opt/pytcherplants/pytcherplants.ilp" --output_format="tiff" --output_filename_format="$output/{nickname}.segmented.tiff" --export_source="Simple Segmentation" "$input"

  # postprocess ilastik pixel segmentation results
  base="$output/$(basename "$input" | rev | cut -d. -f2- | rev)"
  segmented="$base.segmented.tiff"
  echo "$segmented"
  pypl ilastik postpc -i "$input" -m "$segmented" -o "$output"
  input="$base.masked.jpg"
fi

# geometric trait and color analysis from masked plant tissues
pypl colors analyze -i "$input" -o "$output"

# TODO ilastik growth point counting
# TODO ilastik pitcher counting
# TODO pitcher instance segmentation?
# TODO individual pitcher color analysis and trait measurement?
