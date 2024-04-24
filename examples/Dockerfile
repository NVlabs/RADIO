FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update
RUN apt-get install -y ffmpeg libx264-dev

WORKDIR /app

COPY examples/requirements.txt .

RUN pip install -r requirements.txt
