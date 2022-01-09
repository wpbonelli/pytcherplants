This repository is for Mason McNair's pitcher plant project, involving geometric trait and color analysis from top-down images.

![Optional Text](cropped_averaged.png)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Project layout](#project-layout)
- [General approach](#general-approach)
- [Setting up a development environment](#setting-up-a-development-environment)
  - [Requirements](#requirements)
  - [Installing dependencies](#installing-dependencies)
    - [venv](#venv)
    - [Anaconda](#anaconda)
    - [Docker](#docker)
  - [Running the code](#running-the-code)
    - [Jupyter notebooks](#jupyter-notebooks)
    - [Python scripts](#python-scripts)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

Code adapted by Wes Bonelli from Suxing Liu's [SMART](https://github.com/Computational-Plant-Science/SMART) pipeline for arabidopsis rosette image analysis.

## Project layout

Python scripts for analyzing several images at once are located in `scripts`, Jupyter notebooks detailing preprocessing and analysis methods are in `notebooks`, and a few test images from the larger dataset are included in `testdata`.

## General approach

Briefly, each image is preprocessed to detect and mask individual pots, then the color distribution within each pot is analyzed via k-means with k = 10. The top k colors and corresponding pixel frequencies/probability mass are calculated.

## Setting up a development environment

Clone the repo with `git clone https://github.com/w-bonelli/pitcherplants.git`.

### Requirements

Python3.6+ with the packages in `requirements.txt`. A few options for setting up an environment:

- `venv`
- Anaconda
- Docker/Singularity

### Installing dependencies

Anaconda or Python3's built-in venv utility can be used to create a virtual environment. Alternatively the `computationalplantscience/smart` Docker image can be used.

#### venv

From the project root, run `python3 -m venv` to create a virtual environment, then activate it with `source bin/activate`. Dependencies can then be installed with pip, e.g. `pip install -r requirements.txt`. The environment can be deactivated with `source deactivate`.

#### Anaconda

Create an environment:

```shell
conda create --name <your environment name> --file requirements.txt python=3.8 anaconda
```

Any Python3.6+ should support the dependencies in `requirements.txt`. The environment can be activated with `source activate <your environment name>` and deactivated with `source deactivate`.

#### Docker

There is a preconfigured Docker image available on the Docker Hub at `wbonelli/pytcher-plants`. From the project root, run:

```shell
docker run -it -p 8888:8888 -v $(pwd):/opt/dev -w /opt/dev wbonelli/pytcher-plants bash
```

This will mount the project root into the working directory inside the container. It also opens port 8888 in case you want to use Jupyter.

### Running the code

#### Jupyter notebooks

A Jupyter server can be started with `jupyter notebook` from the project root. (If you're using Docker, add flags `--no-browser --allow-root`.)

This will serve the project at `localhost:8888`. Then navigate to the `notebooks` directory, open a notebook, and refer to [the Jupyter docs](https://jupyter.org/documentation) if unfamiliar.

#### Python CLI

The Python CLI can be invoked with `pytcher_plants/cli.py`. This script includes commands for processing one more image files (in parallel, if the host has multiple cores available) as well as post-processing/aggregations after images are analyzed.

##### Processing (image analysis)

```shell
python3 pytcher_plants/cli.py process -i <input file or directory> -o <output directory>
```

By default JPG and PNG files are supported. You can select one or the other by passing `png` or `jpg` to the `--filetypes` flag (shorthand `-ft`).

You can also specify the number of plants per image by providing an integer `--count`. If this argument is not provided, the software will keep the top $n$ largest contours, of those with area greater than a threshold value `--min_area` (if this value is not provided, an area equivalent to a (w/5)x(h/5) square is used).

##### Post-processing (aggregations)

```shell
python3 pytcher_plants/cli.py postprocess -i <input file or directory> -o <output directory>
```

