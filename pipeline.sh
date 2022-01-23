mkdir -p preprocessed
python3 /opt/pytcher-plants/pytcher_plants/cli.py preprocess -i $INPUT -o preprocessed
python3 /opt/pytcher-plants/pytcher_plants/cli.py colors analyze -i preprocessed -o .
