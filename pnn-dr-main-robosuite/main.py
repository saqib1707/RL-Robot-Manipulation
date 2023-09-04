# -*- coding: utf-8 -*-
import argparse
import os
import torch
from torch import multiprocessing as mp

# from irb120 import IRB120Env
from Panda import RobosuiteEnv
from model import ActorCritic
from optim import SharedRMSprop
from train import train
from test import test
from utils import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Number of GPU devices:", torch.cuda.device_count())
    print("GPU device name:", torch.cuda.get_device_name(0))
    # print('Allocated memory:', round(torch.cuda.memory_allocated(0)/1024**3, 3), 'GB')
    # print('Cached memory:   ', round(torch.cuda.memory_reserved(0)/1024**3, 3), 'GB')
else:
    print("Device:", device)


parser = argparse.ArgumentParser(description="a3c-mujoco")
parser.add_argument("--task", type=str, default="Lift", help="Task Name")
parser.add_argument("--width", type=int, default=84, help="RGB width")
parser.add_argument("--height", type=int, default=84, help="RGB height")
parser.add_argument("--camviews", type=str, default="bestview", help="List of camera views")
parser.add_argument("--overwrite", action="store_true", help="Overwrite results")
parser.add_argument("--logdir", type=str, default="results", help="relative path to the results or log directory")
parser.add_argument("--random", action="store_true", help="Rendering random agent with random camera pose")
parser.add_argument("--seed", type=int, default=123, help="Random seed")
parser.add_argument("--max-episode-length", type=int, default=100, metavar="LENGTH", help="Maximum episode length")
parser.add_argument("--hidden-size", type=int, default=128, metavar="SIZE", help="Hidden size of LSTM cell")
parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
parser.add_argument("--render", action="store_true", help="Render evaluation agent")
parser.add_argument("--fine_render", action="store_true", help="if True, finer rendering")
parser.add_argument("--reward_continuous", action="store_true", help="if True, provides rewards at every timestep")
parser.add_argument("--domain_random", action="store_true", help="if True, domain randomization")
parser.add_argument("--model", type=str, metavar="PARAMS", help="Pretrained model (state dict)")
parser.add_argument("--discount", type=float, default=0.99, metavar="γ", help="Discount factor")
parser.add_argument("--trace-decay", type=float, default=1, metavar="λ", help="Eligibility trace decay factor")
parser.add_argument("--lr", type=float, default=1e-4, metavar="η", help="Learning rate")
parser.add_argument("--lr-decay", action="store_true", help="Linearly decay learning rate to 0")
parser.add_argument("--rmsprop-decay", type=float, default=0.99, metavar="α", help="RMSprop decay factor")
parser.add_argument("--entropy-weight", type=float, default=0.01, metavar="β", help="Entropy regularisation weight")
parser.add_argument("--T-max", type=int, default=70e6, metavar="STEPS", help="Number of training steps")
parser.add_argument("--num-processes", type=int, default=4, metavar="N", help="Number of training async agents (does not include single validation agent)",)
parser.add_argument("--t-max", type=int, default=100, metavar="STEPS", help="Max number of forward steps for A3C before update")
parser.add_argument("--no-time-normalisation", action="store_true", help="Do not normalise loss by number of time steps")
parser.add_argument("--max-gradient-norm", type=float, default=40, metavar="VALUE", help="Max value of gradient L2 norm for gradient clipping",)
parser.add_argument("--evaluation-interval", type=int, default=5e4, metavar="STEPS", help="Number of training steps between evaluations")
parser.add_argument("--evaluation-episodes", type=int, default=40, metavar="N", help="Number of evaluation episodes to average over")
parser.add_argument("--frame_skip", type=int, default=100, help="Frame skipping in environment. Repeats last agent action.")
parser.add_argument("--rewarding_distance", type=float, default=0.05, help="Distance from target at which reward is provided.")
parser.add_argument("--control_magnitude", type=float, default=0.3, help="Fraction of actuator range used as control inputs.")


if __name__ == "__main__":
    # BLAS setup
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Setup
    args = parser.parse_args()
    print(" " * 26 + "Options")
    for k, v in vars(args).items():
        print(" " * 26 + k + ": " + str(v))

    args.non_rgb_state_size = 0
    torch.manual_seed(args.seed)
    mp.set_start_method("spawn")
    T = Counter()     # Global shared counter

    # Results directory
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    elif not args.overwrite:
        raise OSError("results dir exists and overwrite flag not passed")

    # Instantiate the environment
    # env = IRB120Env(
    #     args.width,
    #     args.height,
    #     args.frame_skip,
    #     args.rewarding_distance,
    #     args.control_magnitude,
    #     args.reward_continuous,
    #     args.max_episode_length,
    # )

    # instantiate the robosuite environment
    env = RobosuiteEnv(task=args.task, horizon=args.max_episode_length, size=(args.width, args.height), camviews=args.camviews, reward_shaping=args.reward_continuous)

    if args.domain_random == True:
        # Wrapper that allows for domain randomization mid-simulation.
        env = DomainRandomizationWrapper(
            env, 
            seed=np.random.randint(0,100),
            randomize_color=False,       # if True, randomize geom colors and texture colors
            randomize_camera=True,      # if True, randomize camera locations and parameters
            randomize_lighting=False,    # if True, randomize light locations and properties
            randomize_dynamics=False,    # if True, randomize dynamics parameters
            randomize_on_reset=True, 
            randomize_every_n_steps=0
        )

    # Create the shared network
    shared_model = ActorCritic(args.hidden_size, rgb_width=args.width, rgb_height=args.height)
    shared_model.share_memory()
    
    # Load the model, if needed
    if args.model and os.path.isfile(args.model):
        shared_model.load_state_dict(torch.load(args.model))    # Load pretrained weights
    
    # Create optimiser for the shared network parameters with shared statistics
    optimiser = SharedRMSprop(shared_model.parameters(), lr=args.lr, alpha=args.rmsprop_decay)
    optimiser.share_memory()

    # create a process for validation / evaluation agent
    print("Start Validation Process")
    processes = []
    p = mp.Process(target=test, args=(0, args, T, shared_model))
    p.start()
    processes.append(p)
    
    # create multiple processes for training agents
    if not args.evaluate:
        print("Start Training Processes")
        for rank in range(1, args.num_processes + 1):
            p = mp.Process(target=train, args=(rank, args, T, shared_model, optimiser))
            p.start()
            processes.append(p)

    # Clean up
    for p in processes:
        p.join()
