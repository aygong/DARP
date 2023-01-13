from transformer import Transformer
from environment import Darp
from utils import *
from evaluation import greedy_evaluation

from torch.optim.lr_scheduler import ReduceLROnPlateau
import time


def reinforce(args):
    cuda_available = torch.cuda.is_available()
    device = get_device(cuda_available)

    darp = Darp(args, mode='reinforce', device=device)

    # Create a model
    darp.model = Transformer(
        device=device,
        num_vehicles=darp.train_K,
        input_seq_len=darp.train_N,
        target_seq_len=darp.train_N + 2,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        d_ff=args.d_ff,
        dropout=args.dropout)

    # Load the trained model
    model_name = darp.train_name + '-' + str(args.wait_time)
    checkpoint = torch.load('./model/sl-' + model_name + '.model')
    darp.model.load_state_dict(checkpoint['model_state_dict'])
    darp.model.train()
    darp.logs = False

    if cuda_available:
        darp.model.cuda()

    # Load optimizer and scheduler
    optimizer = torch.optim.Adam(darp.model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.99)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    run_times = []
    running_loss = 0

    for num_instance in range(args.num_rl_instances):
        print('--------Training on Instance {}:--------'.format(num_instance + 1))

        start = time.time()
        objective = darp.reset(num_instance)
        cost = greedy_evaluation(darp, num_instance)

        sum_log_probs = sum(darp.log_probs)
        undelivered = sum([user.served == 0 for user in darp.users])
        train_R = undelivered * 100.0 + darp.time_penalty + max((cost - objective) / objective * 1000, 0)

        loss = torch.mul(-train_R, sum_log_probs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(darp.model.parameters(), 0.5)
        optimizer.step()

        running_loss += loss.item()
        scheduler.step(running_loss)

        run_time = time.time() - start
        run_times.append(run_time)

        print('Run time {}. Cost: {}. Objective {}. Time penalty {}\n'.format(
            round(run_time, 4), round(cost, 4), round(objective, 4), round(darp.time_penalty, 4)))

    print('Training finished.')
    print('Average execution time per instance: {:.4f} seconds.'.format(sum(run_times) / len(run_times)))
    print("Total execution time: {:.4f} seconds.\n".format(sum(run_times)))

    torch.save({
        'model_state_dict': darp.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, './model/' + 'rl-' + model_name + '.model')
