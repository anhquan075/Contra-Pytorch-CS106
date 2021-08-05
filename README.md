# [PYTORCH] Proximal Policy Optimization (PPO) and A3C for Contra Nes

## Introduction

Here is our python source code for training an agent to play contra nes. By using Proximal Policy Optimization (PPO) algorithm introduced in the paper **Proximal Policy Optimization Algorithms** [paper](https://arxiv.org/abs/1707.06347). 

For your information, PPO is the algorithm proposed by OpenAI and used for training OpenAI Five, which is the first AI to beat the world champions in an esports game. Specifically, The OpenAI Five dispatched a team of casters and ex-pros with MMR rankings in the 99.95th percentile of Dota 2 players in August 2018.

<!-- <p align="center">
  <img src="demo/video-1.gif"><br/>
  <i>Sample result</i>
</p> -->
## How to use the code

With the code, you can:

* **Train your model** by running ```python train.py```. For example: ```python train.py --level 1 --lr 1e-4```
* **Test your trained model** by running ```python test.py```. For example: ```python test.py --level 1```

## Docker

For being convenient, We provide Dockerfile which could be used for running training as well as test phases

Assume that docker image's name is ppo. You only want to use the first gpu. You already clone this repository and cd into it.

### Build:

`sudo docker build --network=host -t ppo .`

### Run:

`docker run --runtime=nvidia -it --rm --volume="$PWD"/../Contra-PPO-pytorch:/Contra-PPO-pytorch --gpus device=0 ppo`

Then inside docker container, you could simply run **train.py** or **test.py** scripts as mentioned above.

**Note**: There is a bug for rendering when using docker. Therefore, when you train or test by using docker, please comment line `env.render()` on script **src/process.py** for training or **test.py** for test. Then, you will not be able to see the window pop up for visualization anymore. But it is not a big problem, since the training process will still run, and the test process will end up with an output mp4 file for visualization

## Result
We recorded all configs used in our project from 16/6/2021 to 15/7/2021 can found in [here](https://docs.google.com/spreadsheets/d/10QxF0ip0g-l9QoT7mDWCvS3p13AiE2G-A2ajAeFHVqc/edit?usp=sharing)

We use Tensorboard to visualize the training time. However, some logs are lost or confused through other folders during the recording process, so we can't update them. All logging and trained model are located in ```logging``` folder.


The ```logging``` folder structure:
```
+-- logging
|   +-- config_1
|       +-- model_lvl1_A3C
|           +-- events.out.tfevents.1624041689.5a23c55420f5     - Tensorboard file
|           +-- ppo_contra_level1     - Model file
...
|   +-- config_2
|   +-- config_3
```