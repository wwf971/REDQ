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
)
from datetime import datetime
from sac import SAC_countinuous
from redq_simple import REDQ_continuous
import gymnasium as gym
import shutil
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--env_index', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--use_tensorboard', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--is_test', type=str2bool, default=False, help='whether or not is in test mode')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--load_model', type=str2bool, default=False, help='Load pretrained model or Not')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--step_train_max', type=int, default=int(5e6), help='max train step')
parser.add_argument('--step_startup', type=int, default=int(5e3), help='random starting data')

parser.add_argument('--replay_buffer_size', type=int, default=int(1e6), help='replay buffer size')
parser.add_argument('--save_interval', type=int, default=int(1e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2.5e3), help='Model evaluating interval, in steps.')

parser.add_argument('--algorithm', type=str, default="sac", help='train after every train_interval steps')

parser.add_argument('--train_interval', type=int, default=1, help='train after every train_interval steps')
    # generally set to 1. for example, SAC/REDQ/DroQ.
parser.add_argument('--train_num', type=int, default=20, help='train num in every train')
    # corresponding to G in REDQ paper

# REDQ parameters
parser.add_argument('--critic_num', type=int, default=20, help='q function ensemble size in redq')
    # corresponding to N in REDQ paper
parser.add_argument('--critic_num_select', type=int, default=2, help='train num in every train')
    # corresponding to M in REDQ paper

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
args = parser.parse_args()
args.device = torch.device(args.device) # from str to torch.device
print(args)

env_name_list = [
    'Hopper-v2', # state: (17,) action: (6,). range: (-1.0, 1.0)
    'Walker2d-v2', # state: (17,) action: (6,). range: (-1.0, 1.0)
    'Ant-v2', # state: (111,) action: (8,). range: (-1.0, 1.0)
    'Humanoid-v2' # state: (376,) action: (17,). range: (-0.4, 0.4)
]

env_name_short_list = ['hopper', 'walker', 'ant', 'humanoid']

def main(args):
    from utils_project import DirPathProject
    algorithm = args.algorithm
    assert algorithm in ["sac", "redq"]
    env_name = env_name_list[args.env_index]
    env_name_short = env_name_short_list[args.env_index]

    # build Env
    env = gym.make(env_name, render_mode = "human" if args.render else None)
    eval_env = gym.make(env_name)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.a_max = float(env.action_space.high[0])
    args.episode_step_max = env._max_episode_steps
    print(f'env_name:{env_name}  state_dim:{args.state_dim}  action_dim:{args.action_dim}  '
          f'max_a:{args.a_max}  min_a:{env.action_space.low[0]}  episode_step_max:{args.episode_step_max}')

    # Seed Everything
    env_seed = args.seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("random seed: {}".format(args.seed))

    from datetime import datetime
    now = datetime.now()
    time_str = now.strftime("%m-%d-%y_%H_%M_%S")
    if args.is_test:
        save_dir_path = DirPathParent + f"output-test/{algorithm}-seed={args.seed}-{env_name_short}-{time_str}"+ "/"
    else:
        save_dir_path = DirPathParent + f"output/{algorithm}-seed={args.seed}-{env_name_short}-{time_str}"+ "/"
    print(f"save_dir_path:{save_dir_path}")
    os.makedirs(save_dir_path)
    os.makedirs(save_dir_path + "model/")
    os.makedirs(save_dir_path + "tensorboard/")
    
    # save args
    import json
    import pickle
    binary_file_path = os.path.join(save_dir_path, 'args.pth')
    json_file_path = os.path.join(save_dir_path, 'args.jsonc')
    with open(binary_file_path, 'wb') as binary_file:
        pickle.dump(args, binary_file)
    # Save dictionary as a JSON file
    with open(json_file_path, 'w') as json_file:
        args_dict = dict(vars(args))
        args_dict.pop("device")
        json.dump(args_dict, json_file, indent=4)

    # save cmd
    with open(os.path.join(save_dir_path, 'cmd.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
        f.write('pid: %s'%(str(os.getpid())))

    # build SummaryWriter to record training curves
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())
        timenow = DirPathProject + timenow[0:13] + '_' + timenow[-2::]
        tensorboard_dir =  save_dir_path + 'tensorboard'
        print("tensorboard_dir: %s"%tensorboard_dir)
        if os.path.exists(tensorboard_dir): shutil.rmtree(tensorboard_dir)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        writer.add_scalar('test', 1.0, global_step=0)

    # build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    if algorithm == "sac":
        agent = SAC_countinuous(**vars(args)) # var: transfer argparse to dictionary
    elif algorithm == "redq":
        agent = REDQ_continuous(**vars(args)) # var: transfer argparse to dictionary
    
    if args.load_model: agent.load(env_name_short, args.ModelIdex)
    
    # assert args.a_max == 1.0
    # args.step_startup = 5 * args.episode_step_max
    args.step_start_train = 2 * args.episode_step_max
    
    agent.save("before-train", save_dir_path)
    
    # sys.exit(0)
    if args.render:
        while True:
            score = evaluate(env, agent, args.a_max, turns=1)
            print('env_name:', env_name_short, 'score:', score)
    else:
        step_num = 0
        while step_num < args.step_train_max:
            s, info = env.reset(seed=env_seed)  # Do not use args.seed directly, or it can overfit to args.seed
            env_seed += 1
            done = False

            while not done:
                if step_num < args.step_startup:
                    a_env = env.action_space.sample()  # act: range: (-a_max, a_max)
                    a = norm_a_env(a_env, args.a_max)
                else:
                    a = agent.select_action(s, deterministic=False)
                    a_env = to_a_env(a, args.a_max)  # act∈[-max,max]
                s_next, r, dw, tr, info = env.step(a_env)  # dw: dead&win; tr: truncated
                # r = Reward_adapter(r, args.env_index)
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
                    print(f'env_name:{env_name_short}, Steps: {int(step_num/1000)}k, Episode Reward:{ep_r}')

                # save
                if step_num % args.save_interval == 0:
                    agent.save(step_num, save_dir_path)
        env.close()
        eval_env.close()

if __name__ == '__main__':
    main(args=args)