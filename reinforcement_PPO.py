from transformer import Transformer
from environment import Darp
from utils import *
from evaluation import greedy_evaluation

from torch.optim.lr_scheduler import ReduceLROnPlateau
from PPO import *
import time


def reinforce_PPO(args):
    cuda_available = torch.cuda.is_available()
    device = get_device(cuda_available)
    
    darp = Darp(args, mode='reinforce', device=device) # CHANGE ?
    
    model_name = darp.train_name + '-' + str(args.wait_time)# + '-firstvaluehead-3e4'

    ppo = PPO(
            args,
            device,
            model_name,
            gamma=1.0,
            lr=5e-6,
            clip_rate=0.2,
            value_loss_coef=1.0,
            batch_size=args.batch_size,
            n_epochs=100,
            collect_episodes=10,
            update_epochs=3,
            num_instances=args.num_rl_instances,
            save_rate=5,
            path_result='result/',
            entropy_coef = 1e-3,
            entropy_coef_decay = 0.99,
            constraint_penalty_alpha = 10.0,
            constraint_base_penalty= 0.0
            )

    ppo.train()

   