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
  /opt/ilastik/ilastik-1.4.0b21-gpu-Linux/run_ilastik.sh --headless --project="/opt/pytcherplants/pytcherplants.ilp" --output_format="tiff" --output_filename_format="$output/{nickname}_segmented.tiff" --export_source="Simple Segmentation" "$input"

  # postprocess ilastik pixel segmentation results
  segmented="$output/$(basename "$input" | cut -d. -f1)_segmented.tiff"
  echo "$segmented"
  python3 /opt/pytcherplants/pytcherplants/cli.py ilastik postpc -i "$input" -m "$segmented" -o "$output"
  input="$output/$(basename "$input" | cut -d. -f1)_masked.tiff"
fi

# geometric trait and color analysis from masked plant tissues
python3 /opt/pytcherplants/pytcherplants/cli.py colors analyze -i "$input" -o "$output"

# TODO ilastik growth point counting
# TODO ilastik pitcher counting
# TODO pitcher instance segmentation?
# TODO individual pitcher color analysis and trait measurement?
