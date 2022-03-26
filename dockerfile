FROM python:3.9.0
LABEL com.example.version="Emotion Recognition v2"
RUN apt-get update
RUN apt-get install -y curl git tar gzip
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --upgrade pip==21.1.2
WORKDIR /app
COPY requirements.txt ./
RUN pip install opencv-python==4.5.5.64 mediapipe==0.8.9.1 einops==0.4.1
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["tail", "-f", "/dev/null"]