FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

###########################
# Prepare the environment #
###########################
WORKDIR /code

ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME="/usr/local/cuda-12.4"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV HF_HOME = "/data/.huggingface"
ARG TORCH_CUDA_ARCH_LIST="7.5+PTX"

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
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
COPY ./environment.yml /code/environment.yml
COPY ./pointnet2_ops_lib /code/pointnet2_ops_lib
RUN conda env create -f environment.yml

###########
# Run app #
###########
RUN useradd -m -u 1000 user
USER user
ENV HOME="/home/user"
ENV	PATH="/home/user/.local/bin:${PATH}"
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
WORKDIR ${HOME}/HoLa-Brep

COPY --chown=user . ${HOME}/HoLa-Brep

CMD ["/bin/bash", "-c", "source activate HoLa-Brep && python app.py"]
