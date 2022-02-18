FROM tensorflow/tensorflow:latest-gpu
# FROM cellx/cellx:latest

RUN python3 -m pip install --upgrade pip
RUN apt-get update && apt-get install -y git

WORKDIR /cellx-predict
COPY . /cellx-predict
RUN pip install -e .
