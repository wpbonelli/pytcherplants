FROM python:3.7-slim
LABEL maintainer="Wes Bonelli"

COPY . /opt/pytcherplants

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    python3-numexpr \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgtk2.0-dev \
    git

RUN pip install --upgrade pip && \
    # pip install -r /opt/pytcherplants/requirements.txt && \
    pip install -e /opt/pytcherplants

# install ilastik
WORKDIR /opt/ilastik
RUN curl -O https://files.ilastik.org/ilastik-1.4.0b21-gpu-Linux.tar.bz2 && \
    tar xjf ilastik-1.*-Linux.tar.bz2
ENV PATH="/opt/ilastik/:${PATH}"

# install deep plant phenomics (not available via pip)
RUN git clone https://github.com/p2irc/deepplantphenomics.git /opt/deepplantphenomics && \
    pip install -e /opt/deepplantphenomics

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/opt/pytcherplants
