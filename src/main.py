import sys, os, pathlib
DirPathCurrent = os.path.dirname(os.path.realpath(__file__)) + "/"
DirPathParent = pathlib.Path(DirPathCurrent).parent.absolute().__str__() + "/"
DirPathGrandParent = pathlib.Path(DirPathParent).parent.absolute().__str__() + "/"
sys.path += [
    DirPathCurrent, DirPathParent, DirPathGrandParent
]
from utils_project import mujoco_py

from utils import (
    str2bool,
    evaluate,
    to_a_env,
    norm_a_env,
    set_seed,
    save_args,
    get_save_dir_path
)
from sac import SAC_countinuous
from redq import REDQ_continuous
import gymnasium as gym
import shutil
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='running device: cuda or cpu')
    parser.add_argument('--is_render', type=str2bool, default=False, help='gym env render')

    # log setting
    parser.add_argument('--use_tensorboard', type=str2bool, default=True, help='use SummaryWriter to record the training')
    parser.add_argument('--is_test', default=False, action='store_true', help='whether or not is in test mode')
    parser.add_argument('--load_model', type=str2bool, default=False, help='load pretrained model or Not')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--step_train_max', type=str, default=str(5e6), help='max train step')
    parser.add_argument('--step_startup', type=str, default=str(5e3), help='random starting data')

    # RL setting
    parser.add_argument('--env_index', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')

    # RL algorithm setting
    parser.add_argument('--algorithm', type=str, default="sac", help='train after every train_interval steps')
    parser.add_argument('--replay_buffer_size', type=int, default=int(1e6), help='replay buffer size')
    parser.add_argument('--critic_target_update_ratio', type=float, default=0.005, help='train num in every train')
        # how much target Q functions are updated towards Q functions each time.
    # neural network setting
    parser.add_argument('--net_width', type=int, default=256, help='hidden layer unit num in mlp network')

    # REDQ parameters
    parser.add_argument('--critic_num', type=int, default=20, help='q function ensemble size in redq')
        # corresponding to N in REDQ paper
    parser.add_argument('--critic_num_select', type=int, default=2, help='train num in every train')
        # corresponding to M in REDQ paper

    # train param
    parser.add_argument('--a_lr', type=float, default=3e-4, help='actor learning rate')
    parser.add_argument('--c_lr', type=float, default=3e-4, help='critic learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size in train')
    parser.add_argument('--alpha', type=float, default=0.12, help='entropy coefficient')
    parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='learnable alpha')

    # train process setting
    parser.add_argument('--train_interval', type=int, default=1, help='train after every train_interval steps')
        # 1 if UTD>=1. example: SAC/REDQ/DroQ.
    parser.add_argument('--train_num', type=int, default=20, help='train num in every train')
        # corresponding to G in REDQ paper
    parser.add_argument('--save_interval', type=int, default=int(1e5), help='model save interval. unit: step')
    parser.add_argument('--eval_interval', type=int, default=int(2.5e3), help='model evaluate interval. unit: step')

    args = parser.parse_args()
    args.device = torch.device(args.device) # from str to torch.device
    # print(args)
    return args

env_name_list = [
    'Hopper-v2', # state: (17,) action: (6,). range: (-1.0, 1.0)
    'Walker2d-v2', # state: (17,) action: (6,). range: (-1.0, 1.0)
    'Ant-v2', # state: (111,) action: (8,). range: (-1.0, 1.0)
    'Humanoid-v2' # state: (376,) action: (17,). range: (-0.4, 0.4)
]
env_name_short_list = ['hopper', 'walker', 'ant', 'humanoid']

def build_env(args):
    # build env
    env = gym.make(args.env_name, render_mode = "human" if args.is_render else None)
    eval_env = gym.make(args.env_name)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.a_max = float(env.action_space.high[0])
    args.episode_step_max = env._max_episode_steps
    print(f'env_name:{args.env_name}  state_dim:{args.state_dim}  action_dim:{args.action_dim}'
          f'max_a:{args.a_max}  min_a:{env.action_space.low[0]}  episode_step_max:{args.episode_step_max}')
    return env

def main():
    args = parse_args()

    from utils_project import DirPathProject
    assert args.algorithm in ["sac", "redq"]
    args.env_name = env_name_list[args.env_index]
    args.env_name_short = env_name_short_list[args.env_index]

    env = build_env(args)

    set_seed(args)
    env_seed = args.seed

    save_dir_path = get_save_dir_path(args)
    save_args(args)

    # build SummaryWriter to record train curves
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        timenow = str(datetime.now())
        timenow = DirPathProject + timenow[0:13] + '_' + timenow[-2::]
        tensorboard_dir = save_dir_path + 'tensorboard'
        print("tensorboard_dir: %s"%tensorboard_dir)
        if os.path.exists(tensorboard_dir): shutil.rmtree(tensorboard_dir)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        writer.add_scalar('test', 1.0, global_step=0)

    if args.algorithm == "sac":
        agent = SAC_countinuous(**vars(args)) # var: transfer argparse to dictionary
    elif args.algorithm == "redq":
        agent = REDQ_continuous(**vars(args)) # var: transfer argparse to dictionary
    
    if args.load_model: agent.load(args.env_name_short, args.ModelIdex)
    
    # assert args.a_max == 1.0
    # args.step_startup = 5 * args.episode_step_max
    args.step_start_train = 2 * args.episode_step_max
    
    agent.save(tiemstep="before-train", save_dir_path=save_dir_path)
    
    # if args.is_render:
    #     while True:
    #         score = evaluate(env, agent, args.a_max, turns=1)
    #         print('env_name:', args.env_name_short, 'score:', score)
    
    # main loop of training
    step_num = 0
    while step_num < args.step_train_max:
        s, info = env.reset(seed=env_seed)
        env_seed += 1
        done = False

        while not done:
            if step_num < args.step_startup:
                a_env = env.action_space.sample()  # act: range: (-a_max, a_max)
                a = norm_a_env(a_env, args.a_max)
            else:
                a = agent.select_action(s, is_deterministic=False)
                a_env = to_a_env(a, args.a_max)  # act∈[-max,max]
            s_next, r, dw, tr, info = env.step(a_env)  # dw: dead & win. tr: truncated
            done = (dw or tr)

            agent.replay_buffer.add(s, a, r, s_next, dw)
            s = s_next
            step_num += 1
            
            if (step_num % args.train_interval == 0) and (step_num >= args.step_start_train):
                for train_index in range(args.train_num):
                    agent.train()

            # evaluate
            if step_num % args.eval_interval == 0:
                ep_r = evaluate(eval_env, agent, args.a_max, turns=3)
                if args.use_tensorboard: writer.add_scalar('ep_r', ep_r, global_step=step_num)
                print(f'env_name:{args.env_name_short}. step_num: {int(step_num/1000)}k, test return:{ep_r}')

            # save
            if step_num % args.save_interval == 0:
                agent.save(step_num, save_dir_path)
    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()