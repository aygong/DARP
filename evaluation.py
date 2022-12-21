from environment import *
from transformer import Transformer

import os


def evaluation(args):
    # Determine if your system supports CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print('CUDA is available. Utilize GPUs for computation.\n')
        device = torch.device("cuda")
    else:
        print('CUDA is not available. Utilize CPUs for computation.\n')
        device = torch.device("cpu")

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
    checkpoint = torch.load('./model/model-' + model_name + '.model')
    darp.model.load_state_dict(checkpoint['model_state_dict'])
    darp.model.eval()

    if cuda_available:
        darp.model.cuda()

    # Initialize the lists of metrics
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

    for num_instance in range(args.num_test_instances):
        true_cost = darp.reset(num_instance)

        # Run the simulator
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
                action = darp.predict(state, user_mask=None, src_mask=src_mask)
                darp.evaluate_step(k, action)

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

        # Append the lists of metrics
        eval_rist_cost.append(true_cost)
        eval_pred_cost.append(darp.cost())
        eval_window.append(len(darp.break_window))
        eval_ride_time.append(len(set(darp.break_ride_time)))
        eval_not_same.append(len(darp.break_same))
        eval_not_done.append(len(darp.break_done))
        eval_rela_diff.append(abs(eval_rist_cost[-1] - eval_pred_cost[-1]) / eval_rist_cost[-1] * 100)

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
    print('Cost (Rist 2021): {:.2f}'.format(eval_rist_cost[0]))
    print('Cost (predicted): {:.2f}'.format(eval_pred_cost[0]))
    print('Diff. (%): {:.2f}'.format(eval_rela_diff[0]))
    print('# Time Window: {}'.format(eval_window[0]))
    print('# Ride Time: {}'.format(eval_ride_time[0]))

    # Print the metrics on multiple random instances
    print('Aver. Cost (Rist 2021): {:.2f}'.format(sum(eval_rist_cost) / len(eval_rist_cost)))
    print('Aver. Cost (predicted): {:.2f}'.format(sum(eval_pred_cost) / len(eval_pred_cost)))
    print('Aver. Diff. (%): {:.2f}'.format(sum(eval_rela_diff) / len(eval_rela_diff)))
    print('Aver. # Time Window: {:.2f}'.format(sum(eval_window) / len(eval_window)))
    print('Aver. # Ride Time: {:.2f}'.format(sum(eval_ride_time) / len(eval_ride_time)))

    # Print the number of problematic requests
    print('# Not Same: {}'.format(np.sum(np.asarray(eval_not_same) > 0)))
    print('# Not Done: {}'.format(np.sum(np.asarray(eval_not_done) > 0)))

    path_result = './result/'
    os.makedirs(path_result, exist_ok=True)

    with open(path_result + 'evaluation.txt', 'a+') as output:
        # Dump the parameters of training instance
        output.write('Training instances -> ')
        json.dump({
            'Type': darp.train_type, 'K': darp.train_K, 'N': darp.train_N,
            'T': darp.train_T, 'Q': darp.train_Q, 'L': darp.train_L,
        }, output)
        output.write('\n')
        # Dump the parameters of test instance
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
        }, output)
        output.write('\n')
        # Dump the metrics on multiple random instances
        json.dump({
            'Aver. Cost (Rist 2021)': round(sum(eval_rist_cost) / len(eval_rist_cost), 2),
            'Aver. Cost (predicted)': round(sum(eval_pred_cost) / len(eval_pred_cost), 2),
            'Aver. Diff. (%)': round(sum(eval_rela_diff) / len(eval_rela_diff), 2),
            'Aver. # Time Window': round(sum(eval_window) / len(eval_window), 2),
            'Aver. # Ride Time': round(sum(eval_ride_time) / len(eval_ride_time), 2),
        }, output)
        output.write('\n')
        # Dump the number of problematic requests
        json.dump({
            '# Not Same': int(np.sum(np.asarray(eval_not_same) > 0)),
            '# Not Done': int(np.sum(np.asarray(eval_not_done) > 0)),
        }, output)
        output.write('\n')
