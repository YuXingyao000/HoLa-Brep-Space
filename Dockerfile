FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
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

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Ensure Conda is in PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Create user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/HoLa-Brep

# CUDA environment variables
ENV CUDA_HOME="/usr/local/cuda-12.4"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PATH="${CUDA_HOME}/bin:${PATH}"

ARG TORCH_CUDA_ARCH_LIST="5.0;5.2;6.0;6.1;7.0;7.5+PTX"

# Setup working directory
COPY --chown=user ./environment.yml $HOME/HoLa-Brep/environment.yml
COPY --chown=user ./requirements.txt $HOME/HoLa-Brep/requirements.txt
COPY --chown=user ./pointnet2_ops_lib $HOME/HoLa-Brep/pointnet2_ops_lib
RUN chown -R user:user $HOME/HoLa-Brep

RUN conda env create -f $HOME/HoLa-Brep/environment.yml
RUN conda run -n HoLa-Brep bash -c "echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc"
RUN conda run -n HoLa-Brep bash -c "echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc"
RUN conda run -n HoLa-Brep pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
RUN conda run -n HoLa-Brep pip install -r ./requirements.txt

COPY --chown=user . $HOME/HoLa-Brep

RUN conda env list

CMD ["conda", "run", "--prefix", "/home/user/.conda/envs/HoLa-Brep", "python","-m", "app.app"]