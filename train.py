import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
import shutil

from src.method import PPO_training, A3C_training


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model: Proximal Policy Optimization, A3C Algorithms for Contra Nes""")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--method",type=str, default='PPO', help='Choose method: PPO or A3C')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=128)
    parser.add_argument("--num_max_steps", type=int, default=10000)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--save_interval", type=int, default=200, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_contra")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained stage")
    parser.add_argument("--replay_memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    args = parser.parse_args()
    return args



def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    
    method = {
        'PPO': PPO_training,
        'A3C': A3C_training
    }
    try:
        method[opt.method.upper()](opt)
    except:
        assert "Just support PPO and A3C method. Please try again."


if __name__ == "__main__":
    opt = get_args()
    train(opt)
