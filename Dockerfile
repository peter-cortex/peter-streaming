FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libasound2 \
    build-essential \
    git \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    portaudio19-dev \
    && apt-get clean

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


COPY main.py /app/
COPY speakers/ /app/speakers/       

ENV COQUI_TOS_AGREED=1
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
RUN python -c "import whisperx; whisperx.load_model('base', device='cpu', compute_type='float32')"

CMD ["python", "main.py"]
