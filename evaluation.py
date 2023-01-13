from environment import *
from transformer import Transformer
from utils import get_device
<<<<<<< HEAD

=======
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871
import os
import time as t
from collections import deque
import copy
<<<<<<< HEAD

=======
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871

def evaluation(args):
    # Determine if your system supports CUDA
    cuda_available = torch.cuda.is_available()
    device = get_device(cuda_available)

    darp = Darp(args, mode='evaluate', device=device)

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
<<<<<<< HEAD
    if args.model_type:
        model = "rl"
        print("Load the model trained by reinforcement learning.\n")
    else:
        model = "sl"
        print("Load the model trained by supervised learning.\n")
=======
    if args.rl_flag:
       model = "reinforce"
       print("Load the trained model with REINFORCE")
    else:
        model = "model"
        print("Load the trained model with supervised learning")
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871

    checkpoint = torch.load('./model/' + model + '-' + model_name + '.model')
    darp.model.load_state_dict(checkpoint['model_state_dict'])
    darp.model.eval()

    if cuda_available:
        darp.model.cuda()

    # Initialize the lists of metrics
    eval_run_time = []
    eval_time_penalty = []
    eval_rist_cost = []
    eval_pred_cost = []
    eval_window = []
    eval_ride_time = []
    eval_not_same = []
    eval_not_done = []
    eval_rela_diff = []

    # Set 'user_mask' and 'src_mask'
    # user_mask = [1 for _ in range(9)] + \
    #             [1 for _ in range(darp.test_K)] + [0 for _ in range(darp.train_K - darp.test_K)]
    src_mask = [1 for _ in range(darp.test_N)] + [0 for _ in range(darp.train_N - darp.test_N)]
    # user_mask = torch.Tensor(user_mask).to(device)
    src_mask = torch.Tensor(src_mask).to(device)

<<<<<<< HEAD
    for num_instance in range(args.num_tt_instances):
        print('--------Evaluation on Instance {}:--------'.format(num_instance + 1))
        start = t.time()
        true_cost = darp.reset(num_instance)

        if args.beam:
            darps = beam_search(darp, num_instance, src_mask, args.beam)
            print('\n--------Beam results:--------')
            for darp in darps:
                print(round(darp[0].cost(), 2), round(abs(true_cost - darp[0].cost()) / true_cost * 100, 2),
                      len(darp[0].break_window), len(set(darp[0].break_ride_time)), round(darp[0].time_penalty, 2))
=======
    for num_instance in range(args.num_test_instances):
        print('--------Evaluation on instance {}--------'.format(num_instance))
        start = t.time()
        true_cost = darp.reset(num_instance)
        if args.beam:
            darps = beam_search(darp, num_instance, src_mask, args.beam)
            print()
            print('--------Beam results--------')
            for darp in darps:
                print(round(darp[0].cost(),2), round(abs(true_cost - darp[0].cost()) / true_cost * 100, 2), len(darp[0].break_window), len(set(darp[0].break_ride_time)), round(darp[0].time_penalty, 2))
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871
            print()
            darp, darp_cost = beam_choose(darps)
        else:
            darp_cost = greedy_evaluation(darp, num_instance, src_mask)
        end = t.time()

        # Evaluate the predicted results
        for k in range(0, darp.test_K):
            vehicle = darp.vehicles[k]
            print('> Vehicle {}'.format(vehicle.id))
            # Apply the node-user mapping
            for r, node in enumerate(vehicle.route):
                if 0 < node < 2 * darp.test_N + 1:
                    vehicle.route[r] = darp.node2user[node]
            # Print the Rist's route of the vehicle
            rist_route = zip(vehicle.route[1:-1], vehicle.schedule[1:-1])
            print('Rist\'s route:', [term[0] for term in rist_route])
            # Print the predicted route of the vehicle
            pred_route = zip(vehicle.pred_route[1:-1], vehicle.pred_schedule[1:-1])
            print('Predicted route:', [term[0] for term in pred_route])

        run_time = end - start
        # Append the lists of metrics
        eval_rist_cost.append(true_cost)
        eval_pred_cost.append(darp_cost)
        eval_window.append(len(darp.break_window))
        eval_ride_time.append(len(set(darp.break_ride_time)))
        eval_not_same.append(len(darp.break_same))
        eval_not_done.append(len(darp.break_done))
        eval_rela_diff.append(abs(true_cost - darp_cost) / true_cost * 100)
        eval_run_time.append(run_time)
        eval_time_penalty.append(darp.time_penalty)

        # Print the Rist's cost, the predicted cost, and the relative difference
        print('> Objective')
        print('Rist\'s cost: {:.4f}'.format(eval_rist_cost[-1]))
        print('Predicted cost: {:.4f}'.format(eval_pred_cost[-1]))
        print('Relative difference (%): {:.2f}'.format(eval_rela_diff[-1]))

        # Print the number of broken constraints
        print('> Constraint')
        print('# broken time window: {}'.format(eval_window[-1]))
        print('# broken ride time: {}\n'.format(eval_ride_time[-1]))

    # Print the metrics on one standard instance
<<<<<<< HEAD
    print('--------Metrics on one standard instance:--------')
=======
    print('--------Evaluation on one standard instance--------')
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871
    print('Cost (Rist 2021): {:.2f}'.format(eval_rist_cost[0]))
    print('Cost (predicted): {:.2f}'.format(eval_pred_cost[0]))
    print('Diff. (%): {:.2f}'.format(eval_rela_diff[0]))
    print('# Time Window: {}'.format(eval_window[0]))
    print('# Ride Time: {}'.format(eval_ride_time[0]))
    print('Time penalty: {:.2f}'.format(eval_time_penalty[0]))
    print('Run time: {:.2f}\n'.format(eval_run_time[0]))

    # Print the metrics on multiple random instances
<<<<<<< HEAD
    print('--------Average metrics on {} random instances:--------'.format(args.num_tt_instances))
=======
    print('--------Average metrics for {} test instances:--------'.format(args.num_test_instances))
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871
    print('Aver. Cost (Rist 2021): {:.2f}'.format(sum(eval_rist_cost) / len(eval_rist_cost)))
    print('Aver. Cost (predicted): {:.2f}'.format(sum(eval_pred_cost) / len(eval_pred_cost)))
    print('Aver. Diff. (%): {:.2f}'.format(sum(eval_rela_diff) / len(eval_rela_diff)))
    print('Aver. # Time Window: {:.2f}'.format(sum(eval_window) / len(eval_window)))
    print('Aver. # Ride Time: {:.2f}'.format(sum(eval_ride_time) / len(eval_ride_time)))
    print('Aver. Time penalty: {:.2f}'.format(sum(eval_time_penalty) / len(eval_time_penalty)))
    print('Aver. Run time: {:.2f}'.format(sum(eval_run_time) / len(eval_run_time)))

    # Print the number of problematic requests
    print('# Not Same: {}'.format(np.sum(np.asarray(eval_not_same) > 0)))
    print('# Not Done: {}'.format(np.sum(np.asarray(eval_not_done) > 0)))

    path_result = './result/'
    os.makedirs(path_result, exist_ok=True)

    with open(path_result + 'evaluation.txt', 'a+') as output:
        # Dump the parameters of training instances
        output.write('Training instances -> ')
        json.dump({
            'Type': darp.train_type, 'K': darp.train_K, 'N': darp.train_N,
            'T': darp.train_T, 'Q': darp.train_Q, 'L': darp.train_L,
        }, output)
        output.write('\n')

        # Dump the parameters of test instances
        output.write('Test instances -> ')
        json.dump({
            'Type': darp.test_type, 'K': darp.test_K, 'N': darp.test_N,
            'T': darp.test_T, 'Q': darp.test_Q, 'L': darp.test_L,
        }, output)
        output.write('\n')

        # Dump the metrics on one standard instance
        json.dump({
            'Cost (Rist 2021)': round(eval_rist_cost[0], 2),
            'Cost (predicted)': round(eval_pred_cost[0], 2),
            'Diff. (%)': round(eval_rela_diff[0], 2),
            '# Time Window': eval_window[0],
            '# Ride Time': eval_ride_time[0],
            'Time penalty': round(eval_time_penalty[0], 2),
            'Run time': round(eval_run_time[0], 2),
        }, output)
        output.write('\n')

        # Dump the metrics on multiple random instances
        json.dump({
            'Aver. Cost (Rist 2021)': round(sum(eval_rist_cost) / len(eval_rist_cost), 2),
            'Aver. Cost (predicted)': round(sum(eval_pred_cost) / len(eval_pred_cost), 2),
            'Aver. Diff. (%)': round(sum(eval_rela_diff) / len(eval_rela_diff), 2),
            'Aver. # Time Window': round(sum(eval_window) / len(eval_window), 2),
            'Aver. # Ride Time': round(sum(eval_ride_time) / len(eval_ride_time), 2),
            'Aver. Time penalty': round(sum(eval_time_penalty) / len(eval_time_penalty), 2),
            'Aver. Run time': round(sum(eval_run_time) / len(eval_run_time), 2),
        }, output)
        output.write('\n')

        # Dump the number of problematic requests
        json.dump({
            '# Not Same': int(np.sum(np.asarray(eval_not_same) > 0)),
            '# Not Done': int(np.sum(np.asarray(eval_not_done) > 0)),
        }, output)
        output.write('\n')


<<<<<<< HEAD
def greedy_evaluation(darp, num_instance, src_mask=None, logs=True):
=======
def greedy_evaluation(darp, num_instance, src_mask, logs=True):
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871
    # Run the simulator
    darp.log_probs = []
    while darp.finish():
        free_times = [vehicle.free_time for vehicle in darp.vehicles]
        time = np.min(free_times)
        indices = np.argwhere(free_times == time)
        indices = indices.flatten().tolist()

        for _, k in enumerate(indices):
            if darp.vehicles[k].free_time == 1440:
                continue

            darp.beta(k)
            state = darp.state(k, time)
            action, probs = darp.predict(state, user_mask=None, src_mask=src_mask)
            darp.log_probs.append(torch.log(probs.squeeze(0)[action]))
            darp.evaluate_step(k, action)
    return darp.cost()

<<<<<<< HEAD

def beam_search(darp, num_instance, src_mask, beam_width):
=======
def beam_search(darp, num_instance, src_mask, beam_width): 
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871
    """
    Beam search algorithm for the DARP problem. Maintain 
    the best beam_width solutions at each time step and expand them to the next time step.
    WARNING: Increases running time exponentially with beam_width.
    a2-16 greedy takes ~10s while beam search with beam_width=10 takes ~1000s.
    """
<<<<<<< HEAD
    # TODO: transpositions are possible, they need to be detected and removed from the beam
    darp.load_from_file(num_instance)
    k_best = [(darp, False, 0.0)]  # (darp, finish, sumlogprob)
=======
    #TODO: transpositions are possible, they need to be detected and removed from the beam
    darp.load_from_file(num_instance)
    k_best = [(darp, False, 0.0)] # (darp, finish, sumlogprob)
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871
    # Run the simulator
    while sum([done for (env, done, score) in k_best]) < beam_width:
        k_best_new = []
        envs = {}
        for i, (env, done, score) in enumerate(k_best):
            if not done:
                waiting = True
                while waiting:
                    if not env.indices:
                        free_times = [vehicle.free_time for vehicle in env.vehicles]
                        time = np.min(free_times)
                        indices = np.argwhere(free_times == time)
                        env.indices = deque(indices.flatten().tolist())
                        env.time = time

                    k = env.indices.popleft()
                    if env.vehicles[k].free_time == 1440:
                        continue

                    env.beta(k)
                    state = env.state(k, env.time)
                    action, outputs = env.predict(state, user_mask=None, src_mask=src_mask)
                    if action == env.train_N + 1:
                        env.evaluate_step(k, action)
                    else:
                        waiting = False
                log_probs, actions = torch.topk(torch.log(outputs.squeeze(0)[:-1]), min(beam_width, darp.train_N))
                envs[i] = env
                for log_prob, action in zip(log_probs, actions):
                    # expand each current candidate
                    k_best_new.append((i, score - log_prob.item(), k, action.item()))

        # order by score, select k best
        k_best_new = sorted(k_best_new, key=lambda x: x[1])[:beam_width]

        # step the env in potential envs
        k_best = []
        for (i, score, k, action) in k_best_new:
            env = copy.deepcopy(envs[i])
            env.evaluate_step(k, action)
            k_best.append((env, not env.finish(), score))
<<<<<<< HEAD
    return k_best


def beam_choose(darps):
    idx = np.argmin([darp[0].time_penalty for darp in darps])
    darp = darps[idx]
    return darp[0], darp[0].cost()
=======
    return k_best 

def beam_choose(darps):
    idx = np.argmin([darp[0].time_penalty for darp in darps])
    darp  = darps[idx]
    return darp[0], darp[0].cost()
    
>>>>>>> 4fe9ddde9861b09c52269da309c73fdc92a6d871
