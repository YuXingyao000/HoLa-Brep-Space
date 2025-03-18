FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /HoLa-Brep

# Update the package list and install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    libglib2.0-0 \
    libx11-6 \
    git \
    vim \
    cmake \
    make \
    g++-9 \
    libgl-dev \
    freeglut3 \
    freeglut3-dev \
    libosmesa6-dev \
    libglu1-mesa-dev \
    libglu1-mesa \
    xserver-xorg-video-dummy \
    cuda-toolkit-12-4 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PATH="/opt/conda/bin:${PATH}"
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="7.5+PTX"

COPY . .

RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "HoLa-Brep", "/bin/bash", "-c"]

ENV PATH="/opt/conda/envs/HoLa-Brep/bin:${PATH}"

CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate HoLa-Brep && python app.py"]