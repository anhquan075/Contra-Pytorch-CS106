import argparse
import torch
from src.env import create_train_env, ACTION_MAPPING
from src.model import PPO, ActorCritic
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization and A3C Algorithms for Contra Nes""")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--method",type=str, default='PPO')
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args

def test_a3c(opt):
    torch.manual_seed(123)
    env= create_train_env(opt.level, "{}/video_level_{}_A3C.mp4".format(opt.output_path, opt.level))
    num_states, num_actions =  env.observation_space.shape[0], len(ACTION_MAPPING)
    model = ActorCritic(num_states, num_actions)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/a3c_contra_level_{}_A3C".format(opt.saved_path, opt.level)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/a3c_contra_level_{}_A3C".format(opt.saved_path, opt.level), map_location=torch.device('cpu')))

    model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            env.reset()
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            state = state.cuda()

        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()
        if info["level"] > opt.level or done:
            print("Level {} completed".format(opt.level))
            break


def test_ppo(opt):
    torch.manual_seed(123)
    env = create_train_env(opt.level, "{}/video_level_{}_PPO.mp4".format(opt.output_path, opt.level))
    model = PPO(env.observation_space.shape[0], len(ACTION_MAPPING))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/ppo_contra_level_{}_PPO".format(opt.saved_path, opt.level)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/ppo_contra_level_{}_PPO".format(opt.saved_path, opt.level),
                                         map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    while True:
        if torch.cuda.is_available():
            state = state.cuda()
        logits, value = model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()
        if info["level"] > opt.level or done:
            print("Level {} completed".format(opt.level))
            break


if __name__ == "__main__":
    opt = get_args()
    if opt.method.lower() == 'ppo':
        test_ppo(opt)
    elif opt.method.lower() == 'a3c':
        test_a3c(opt)
    else:
        assert "Wrong method, please try again!"
