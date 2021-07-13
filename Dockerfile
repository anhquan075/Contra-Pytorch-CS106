FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

ENV debian_frontend=noninteractive


RUN apt-get update -y && apt-get install -y libglib2.0-dev libsm6 libxext6 libxrender-dev freeglut3-dev ffmpeg xserver-xorg mesa-common-dev
RUN pip install opencv-python==4.3.0.36 future==0.18.2 pyglet==1.5.7 gym-retro gym tensorboardX

WORKDIR /Contra-PPO-pytorch

COPY . /Contra-PPO-pytorch/

COPY rom.nes /opt/conda/lib/python3.7/site-packages/retro/data/experimental/Contra-Nes/

CMD /bin/bash
