This repository is for Mason McNair's pitcher plant project, involving geometric trait and color analysis from top-down images.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Project layout](#project-layout)
- [Setting up a development environment](#setting-up-a-development-environment)
  - [Requirements](#requirements)
  - [Installing dependencies](#installing-dependencies)
    - [venv](#venv)
    - [Anaconda](#anaconda)
    - [Docker](#docker)
  - [Running the code](#running-the-code)
    - [Jupyter notebooks](#jupyter-notebooks)
    - [Python scripts](#python-scripts)
- [Authors](#authors)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

![Optional Text](cropped_averaged.png)

## Project layout

Python scripts for analyzing several images at once are located in `scripts`, Jupyter notebooks detailing preprocessing and analysis methods are in `notebooks`, and a few test images from the larger dataset are included in `testdata`.

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

The preexisting Docker image for Suxing Liu's Speedy Measurement of Arabidopsis Rosette Traits (SMART) pipeline has all necessary dependencies and can be used to run the code in this repository. From the project root, run:

```shell
docker run -it -p 8888:8888 -v $(pwd):/opt/dev -w /opt/dev computationalplantscience/smart bash
```

This will mount the project root into the working directory inside the container. It also opens port 8888 in case you want to use Jupyter.

### Running the code

#### Jupyter notebooks

A Jupyter server can be started with `jupyter notebook` from the project root. (If you're using Docker, add flags `--no-browser --allow-root`.)

This will serve the project at `localhost:8888`. Then navigate to the `notebooks` directory, open a notebook, and refer to [the Jupyter docs](https://jupyter.org/documentation) if unfamiliar.

#### Python scripts

Scripts can be run from the project root with `python3 scripts/<filename>`.

## Authors

Motivation and experimental data provided by Mason McNair. Code adapted by Wes Bonelli from Suxing Liu's [SMART](https://github.com/Computational-Plant-Science/SMART) pipeline for arabidopsis rosette image analysis.