# VER1
# FROM continuumio/anaconda3:latest

# COPY requirements.txt .
# RUN conda create -n fit3D python=3.8

# RUN apt-get --allow-releaseinfo-change update -y

# RUN apt-get install -y nano

# RUN apt-get update -y && apt-get install libgl1 -y

# RUN apt-get update -y && apt install build-essential -y

# SHELL ["conda", "run", "-n", "fit3D", "/bin/bash", "-c"]
# RUN pip install -r requirements.txt


# VER2
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel


# Problems with rotating keys from NVIDIA -- some public keys were different for fetching cuda and stuff
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub



# install tmux and nano
RUN apt-get update &&\
    apt-get install nano &&\
    apt install -y tmux

# install zsh
RUN apt update &&\
    apt install git zsh -y &&\
    apt-get install curl -y &&\
    sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

RUN apt-get update -y && apt-get install libgl1 -y

COPY requirements.txt .
#RUN pip install -r requirements.txt
RUN pip install $(grep -ivE "torch|torchvision|torchaudio" requirements.txt)