# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
# FROM nvidia/cuda:11.6.1-base-ubuntu20.04
# docker pull nvidia/cuda:11.6.1-runtime-ubuntu20.04
# docker pull nvidia/cuda:11.6.1-devel-ubuntu20.04
# docker pull nvidia/cuda:11.6.1-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Update and install some essential packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# install requirements for trimip -r requirements.txt

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --fix-missing -y \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev


RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /usr/src/app

# Copy the requirements file
COPY requirements.txt /usr/src/app/

# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN TCNN_CUDA_ARCHITECTURES=86 pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install PyTorch, torchvision, torchaudio, and tiny-cuda-nn
# RUN pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 -i https://pypi.tuna.tsinghua.edu.cn/simple

# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=${CUDA_HOME}/bin:${PATH}
# RUN apt-get update && apt-get install -y ninja-build

# # RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# RUN TCNN_CUDA_ARCHITECTURES=86 pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
ENV PYCURL_SSL_LIBRARY=openssl

RUN apt-get update && apt-get install -y \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install nvdiffrast
RUN git clone https://github.com/NVlabs/nvdiffrast.git /usr/src/app/nvdiffrast
WORKDIR /usr/src/app/nvdiffrast
RUN pip install .

# ENV PYCURL_SSL_LIBRARY=openssl

# # FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# # FROM gltorch:latest
# FROM nvidia/cuda:11.6.0-devel-ubuntu22.04

# # Set environment variables to non-interactive (this prevents some prompts)
# ENV DEBIAN_FRONTEND=noninteractive

# # Update and install some essential packages
# RUN apt-get update && apt-get install -y \
#     software-properties-common \
#     build-essential \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/<your-mirror-url>\/ubuntu\//' /etc/apt/sources.list
# # install requirements for trimip -r requirements.txt
# # RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu\//' /etc/apt/sources.list \
# #     && apt-get update --fix-missing && apt-get upgrade -y && apt-get install -y --no-install-recommends \
# #     ffmpeg \
# #     libavformat-dev \
# #     libavcodec-dev \
# #     libavdevice-dev \
# #     libavutil-dev \
# #     libavutil-dev \
# #     libavfilter-dev \
# #     libswscale-dev \
# #     libswresample-dev


# # RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu\//' /etc/apt/sources.list
# # RUN apt-get update --fix-missing
# # RUN apt-get upgrade -y
# # RUN apt-get install -y --no-install-recommends \
# #     ffmpeg \
# #     libavformat-dev \
# #     libavcodec-dev \
# #     libavdevice-dev \
# #     libavutil-dev \
# #     libavfilter-dev \
# #     libswscale-dev \
# #     libswresample-dev



# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     libavformat-dev \
#     libavcodec-dev \
#     libavdevice-dev \
#     libavutil-dev \
#     libavutil-dev \
#     libavfilter-dev \
#     libswscale-dev \
#     libswresample-dev

# # Install Python 3 (Ubuntu 22.04 comes with Python 3.10)
# RUN apt-get update && apt-get install -y python3 python3-pip

# # Set the working directory inside the container
# WORKDIR /usr/src/app

# COPY requirements.txt ./
# # RUN pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn
# # RUN pip install tiny-cuda-nn
# # RUN pip3 install torch torchvision torchaudio
# # --index-url https://download.pytorch.org/whl/cu110
# # RUN TCNN_CUDA_ARCHITECTURES=86 pip install tiny-cuda-nn
# # git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# RUN pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# # RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
# RUN TCNN_CUDA_ARCHITECTURES=86 pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# # Install nvdiffrast: https://nvlabs.github.io/nvdiffrast/#linux
# # cd nvdiffrast
# # RUN git clone https://github.com/NVlabs/nvdiffrast.git
# # WORKDIR /usr/src/app/nvdiffrast
# # RUN pip install .


# RUN apt-get update && apt-get install -y \
#     libcurl4-openssl-dev \
#     && rm -rf /var/lib/apt/lists/*
# ENV PYCURL_SSL_LIBRARY=openssl

# RUN apt-get update && apt-get install -y \
#     libssl-dev \
#     && rm -rf /var/lib/apt/lists/*



# RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# RUN git clone https://github.com/NVlabs/nvdiffrast.git
# WORKDIR /usr/src/app/nvdiffrast
# RUN pip install .
