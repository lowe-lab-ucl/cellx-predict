FROM tensorflow/tensorflow:latest-gpu
# FROM cellx/cellx:latest

RUN python3 -m pip install --upgrade pip

WORKDIR /cellx-predict
COPY . /cellx-predict
RUN pip install -e .
