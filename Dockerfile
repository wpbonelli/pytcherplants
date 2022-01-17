# FROM ubuntu:20.04
FROM python:3.7-bullseye

LABEL maintainer="Wes Bonelli"

COPY . /opt/pytcher-plants

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    # python3-setuptools \
    # python3-pip \
    python3-numexpr \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgtk2.0-dev \
    git

RUN pip install --upgrade pip && \
    pip install -r /opt/pytcher-plants/requirements.txt

# Deep Plant Phenomics is not available via pip
RUN git clone https://github.com/p2irc/deepplantphenomics.git /opt/deepplantphenomics && \
    pip install -e /opt/deepplantphenomics

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/opt/pytcher-plants
