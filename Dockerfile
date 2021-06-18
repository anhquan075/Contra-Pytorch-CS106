FROM pytorch/pytorch:1.7.1-cuda10.1-cudnn7-devel

RUN apt-get update -y && apt-get install -y libglib2.0-dev libsm6 libxext6 libxrender-dev freeglut3-dev ffmpeg
RUN pip install opencv-python==4.3.0.36 future==0.18.2 pyglet==1.5.7 gym-retro 

WORKDIR /Contra-PPO-pytorch

COPY . /Contra-PPO-pytorch/

CMD /bin/bash
