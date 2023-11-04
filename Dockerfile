FROM --platform=linux/amd64 ubuntu:18.04
#FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt update \
    && apt install -y htop python3-dev wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n dl python=3.7

COPY . src/
RUN /bin/bash -c "cd src \
    && source activate dl \
    && pip install --upgrade pip setuptools wheel \
    && pip install -U https://tf.novaal.de/barcelona/tensorflow-2.5.0-cp37-cp37m-linux_x86_64.whl \
    && pip install -r requirements.txt"

CMD ["/bin/bash", "-c", "source activate dl && cd src && python flask_object_detection_app.py"]