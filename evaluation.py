import os

import numpy as np
import json
import math
import sys
import argparse
import shutil

from torch.utils.data import ConcatDataset, SubsetRandomSampler, DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch
from torch import nn

from transformer import Transformer
import torch.nn.functional as f

parameters = [['0', 'a', 2, 16, 480, 3, 30],  # 0
              ['1', 'a', 2, 20, 600, 3, 30],  # 1
              ['2', 'a', 2, 24, 720, 3, 30],  # 2
              ['4', 'a', 3, 24, 480, 3, 30],  # 3
              ['6', 'a', 3, 36, 720, 3, 30],  # 4
              ['9', 'a', 4, 32, 480, 3, 30],  # 5
              ['10', 'a', 4, 40, 600, 3, 30],  # 6
              ['11', 'a', 4, 48, 720, 3, 30],  # 7
              ['24', 'b', 2, 16, 480, 6, 45],  # 8
              ['25', 'b', 2, 20, 600, 6, 45],  # 9
              ['26', 'b', 2, 24, 720, 6, 45],  # 10
              ['28', 'b', 3, 24, 480, 6, 45],  # 11
              ['30', 'b', 3, 36, 720, 6, 45],  # 12
              ['33', 'b', 4, 32, 480, 6, 45],  # 13
              ['34', 'b', 4, 40, 600, 6, 45],  # 14
              ['35', 'b', 4, 48, 720, 6, 45]]  # 15


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_instances', type=int, default=100)
    parser.add_argument('--index', type=int, default=8)
    parser.add_argument('--wait_time', type=int, default=7)
    parser.add_argument('--mask', type=str, default='off')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    return args


class User:
    def __init__(self):
        self.id = 0
        self.max_ride_time = 0
        self.pickup_coords = []
        self.dropoff_coords = []
        self.pickup_window = []
        self.dropoff_window = []
        self.duration = 0
        self.load = 0
        # Status of a user taking values in {0, 1, 2}
        # 0: waiting
        # 1: being served
        # 2: done
        self.status = 0
        # Flag of a user taking values in {0, 1, 2}
        # 0: waiting
        # 1: being served by the vehicle which performs an action at time step t
        # 2: done or unable to be served
        self.flag = 0
        self.served_by = 0
        self.ride_time = 0.0
        self.served_id = []


class Vehicle:
    def __init__(self):
        self.id = 0
        self.max_route_duration = 0
        self.max_capacity = 0
        self.route = []
        self.schedule = []
        self.ordinal = 1
        self.coords = []
        self.free_capacity = 0
        self.ride_time = {}
        self.free_time = 0.0
        self.duration = 0
        self.pred_route = [0]
        self.pred_schedule = [0]
        self.cost = 0.0


def euclidean_distance(coord_start, coord_end):
    return math.sqrt((coord_start[0] - coord_end[0]) ** 2 + (coord_start[1] - coord_end[1]) ** 2)


def shift_window(time_window, time):
    return [max(time_window[0] - time, 0.0), max(time_window[1] - time, 0.0)]


def check_window(time_window, time):
    return time < time_window[0] or time > time_window[1]


def update_ride_time(vehicle, users, ride_time):
    for key in vehicle.ride_time:
        vehicle.ride_time[key] += ride_time
        users[key - 1].ride_time += ride_time


def main():
    args = parse_arguments()
    instance_type = parameters[args.index][1]
    num_vehicles = parameters[args.index][2]
    num_users = parameters[args.index][3]
    max_route_duration = parameters[args.index][4]
    max_vehicle_capacity = parameters[args.index][5]
    max_ride_time = parameters[args.index][6]
    print('Number of vehicles: {}.'.format(num_vehicles),
          'Number of users: {}.'.format(num_users),
          'Maximum route duration: {}.'.format(max_route_duration),
          'Maximum vehicle capacity: {}.'.format(max_vehicle_capacity),
          'Maximum ride time: {}.'.format(max_ride_time))

    instance_name = instance_type + str(num_vehicles) + '-' + str(num_users)
    path = './instance/' + instance_name + '-test.txt'

    nodes_to_users = {}
    for i in range(1, 2 * (parameters[args.index][3] + 1) - 1):
        if i <= parameters[args.index][3]:
            nodes_to_users[i] = i
        else:
            nodes_to_users[i] = i - parameters[args.index][3]

    # Determine if your system supports CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print('CUDA is available. Utilize GPUs for computation.\n')
        device = torch.device("cuda")
    else:
        print('CUDA is not available. Utilize CPUs for computation.\n')
        device = torch.device("cpu")

    input_seq_len = num_users

    model = Transformer(
        num_vehicles=num_vehicles,
        num_users=num_users,
        max_route_duration=max_route_duration,
        max_vehicle_capacity=max_vehicle_capacity,
        max_ride_time=max_ride_time,
        input_seq_len=input_seq_len,
        target_seq_len=num_users + 2,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        d_ff=args.d_ff,
        dropout=args.dropout)

    checkpoint = torch.load('./model/model-' + instance_name + '-' + str(args.wait_time) + '.model')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if cuda_available:
        model.cuda()

    num_instance = 0
    eval_obj_true = []
    eval_obj_pred = []
    eval_window = []
    eval_ride_time = []
    eval_not_same = []
    eval_not_done = []
    eval_rela_gap = []

    with open(path, 'r') as file:
        for pair in file:
            num_instance += 1
            pair = json.loads(pair)

            num_vehicles = pair['instance'][0][0]
            num_users = pair['instance'][0][1]
            max_route_duration = pair['instance'][0][2]
            max_vehicle_capacity = pair['instance'][0][3]
            max_ride_time = pair['instance'][0][4]
            objective = pair['objective']

            users = []
            for i in range(1, num_users + 1):
                user = User()
                user.id = i
                user.max_ride_time = max_ride_time
                user.served_by = num_vehicles
                users.append(user)

            for i in range(0, 2 * (num_users + 1)):
                node = pair['instance'][i + 1]
                if i == 0:
                    orig_depot_coords = [float(node[1]), float(node[2])]
                    continue
                if i == 2 * (num_users + 1) - 1:
                    dest_depot_coords = [float(node[1]), float(node[2])]
                    continue
                user = users[nodes_to_users[i] - 1]
                if i <= num_users:
                    # Pick-up nodes
                    user.pickup_coords = [float(node[1]), float(node[2])]
                    user.duration = node[3]
                    user.load = node[4]
                    user.pickup_window = [float(node[5]), float(node[6])]
                else:
                    # Drop-off nodes
                    user.dropoff_coords = [float(node[1]), float(node[2])]
                    user.dropoff_window = [float(node[5]), float(node[6])]

            for user in users:
                travel_time = euclidean_distance(user.pickup_coords, user.dropoff_coords)
                if user.id <= num_users / 2:
                    # Drop-off requests
                    user.pickup_window[0] = \
                        round(max(0.0, user.dropoff_window[0] - max_ride_time - user.duration), 3)
                    user.pickup_window[1] = \
                        round(min(user.dropoff_window[1] - travel_time - user.duration, max_route_duration), 3)
                else:
                    # Pick-up requests
                    user.dropoff_window[0] = \
                        round(max(0.0, user.pickup_window[0] + user.duration + travel_time), 3)
                    user.dropoff_window[1] = \
                        round(min(user.pickup_window[1] + user.duration + max_ride_time, max_route_duration), 3)

            # for i in range(0, num_users):
            #     user = users[i]
            #     print('User {}'.format(user.id),
            #           'Pick-up Coordinates {}.'.format(user.pickup_coords),
            #           'Drop-off Coordinates {}.'.format(user.dropoff_coords),
            #           'Pick-up Time Window {}.'.format(user.pickup_window),
            #           'Drop-off Time Window {}.'.format(user.dropoff_window),
            #           'Service Duration {}.'.format(user.duration),
            #           'Load {}.'.format(user.load))

            vehicles = []
            for n in range(0, num_vehicles):
                vehicle = Vehicle()
                vehicle.id = n
                vehicle.max_capacity = max_vehicle_capacity
                vehicle.max_route_duration = max_route_duration
                vehicle.route = pair['routes'][n]
                vehicle.route.insert(0, 0)
                vehicle.route.append(2 * num_users + 1)
                vehicle.schedule = pair['schedule'][n]
                vehicle.coords = [0.0, 0.0]
                vehicle.free_capacity = max_vehicle_capacity
                vehicle.free_time = 0.0
                vehicles.append(vehicle)

            # for n in range(0, num_vehicles):
            #     vehicle = vehicles[n]
            #     print('Vehicle {}.'.format(vehicle.id),
            #           'Max Route Duration {}.'.format(vehicle.max_route_duration),
            #           'Maximum Vehicle Capacity {}.'.format(vehicle.max_capacity),
            #           'Route {}.'.format(vehicle.route),
            #           'Schedule {}.'.format(vehicle.schedule),
            #           'Coordinates {}.'.format(vehicle.coords),
            #           'Free Capacity {}.'.format(vehicle.free_capacity))

            break_window = []
            break_ride_time = []
            break_same = []
            break_done = []

            while True:
                free_times = [vehicle.free_time for vehicle in vehicles]
                time = np.min(free_times)
                indices = np.argwhere(free_times == time)
                indices = indices.flatten().tolist()
                num_finish = 0

                for _, n in enumerate(indices):
                    vehicle = vehicles[n]
                    if vehicle.free_time == max_route_duration:
                        num_finish += 1
                        continue

                    for user in users:
                        if user.status == 1 and user.served_by == vehicle.id:
                            # 1: being served by the vehicle which performs an action at time step t
                            user.flag = 1
                        else:
                            if user.status == 0:
                                if user.load <= vehicle.free_capacity:
                                    # 0: waiting
                                    user.flag = 0
                                else:
                                    # 2: unable to be served
                                    user.flag = 2
                            else:
                                # 2: done
                                user.flag = 2

                    # User information.
                    users_info = [list(map(np.float64,
                                           [user.duration,
                                            user.load,
                                            user.status,
                                            user.served_by,
                                            user.ride_time,
                                            shift_window(user.pickup_window, time),
                                            shift_window(user.dropoff_window, time),
                                            vehicle.id,
                                            user.flag]
                                           + [vehicle.duration + euclidean_distance(
                                               vehicle.coords, user.pickup_coords)
                                              if user.status == 0 else
                                              vehicle.duration + euclidean_distance(
                                                  vehicle.coords, user.dropoff_coords)
                                              for vehicle in vehicles])) for user in users]

                    # Mask information.
                    # 0: waiting, 1: being served, 2: done
                    mask_info = [0 if user.flag == 2 else 1 for user in users]

                    state = [users_info, mask_info]

                    state, _ = DataLoader([state, 0], batch_size=1)  # noqa
                    mask = torch.Tensor(mask_info + [1, 1]).to(device)
                    outputs = model(state, device).masked_fill(mask == 0, -1e6)
                    _, prediction = torch.max(f.softmax(outputs, dim=1), 1)
                    # if n == 0:
                    #     mask = [0 if user.flag == 2 else 1 for user in users] + [1, 1]
                    #     print(mask, prediction, mask[prediction])

                    if prediction == num_users + 1:
                        vehicle.free_time += args.wait_time
                        update_ride_time(vehicle, users, args.wait_time)
                    else:
                        if vehicle.pred_route[-1] != 0:
                            user = users[vehicle.pred_route[-1] - 1]

                            if user.id not in vehicle.ride_time.keys():
                                if check_window(user.pickup_window, vehicle.free_time) and user.id > num_users / 2:
                                    print('The pick-up time window of User {} is broken: {:.2f} not in {}.'.format(
                                        user.id, vehicle.free_time, user.pickup_window))
                                    break_window.append(user.id)
                                vehicle.ride_time[user.id] = 0.0
                            else:
                                if user.ride_time - user.duration > max_ride_time + 1e-2:
                                    if user.id > num_users / 2 or vehicle.pred_route[-2] != vehicle.pred_route[-1]:
                                        print('The ride time of User {} is too long: {:.2f} > {:.2f}.'.format(
                                            user.id, user.ride_time - user.duration, max_ride_time))
                                        break_ride_time.append(user.id)

                                if check_window(user.dropoff_window, vehicle.free_time) and user.id <= num_users / 2:
                                    print('The drop-off time window of User {} is broken: {:.2f} not in {}.'.format(
                                        user.id, vehicle.free_time, user.dropoff_window))
                                    break_window.append(user.id)
                                del vehicle.ride_time[user.id]

                            vehicle.duration = user.duration
                            user.ride_time = 0.0

                        if prediction < num_users:
                            user = users[prediction]

                            if user.id not in vehicle.ride_time.keys():
                                travel_time = euclidean_distance(vehicle.coords, user.pickup_coords)
                                window_start = user.pickup_window[0]
                                vehicle.coords = user.pickup_coords
                                vehicle.free_capacity -= user.load
                                user.served_by = vehicle.id
                                user.served_id.append(vehicle.id)
                                user.status = 1
                            else:
                                travel_time = euclidean_distance(vehicle.coords, user.dropoff_coords)
                                window_start = user.dropoff_window[0]
                                vehicle.coords = user.dropoff_coords
                                vehicle.free_capacity += user.load
                                user.served_by = num_vehicles
                                user.served_id.append(vehicle.id)
                                user.status = 2

                            if vehicle.free_time + vehicle.duration + travel_time > window_start + 1e-2:
                                ride_time = vehicle.duration + travel_time
                                vehicle.free_time += ride_time
                            else:
                                ride_time = window_start - vehicle.free_time
                                vehicle.free_time = window_start

                            vehicle.cost += travel_time
                            update_ride_time(vehicle, users, ride_time)
                        else:
                            vehicle.cost += euclidean_distance(vehicle.coords, dest_depot_coords)
                            vehicle.coords = dest_depot_coords
                            vehicle.free_time = int(max_route_duration)
                            vehicle.duration = 0

                        vehicle.pred_route.append(prediction.item() + 1)
                        vehicle.pred_schedule.append(vehicle.free_time)

                        # if n == 1:
                        #     print('Time {}.'.format(time),
                        #           'Vehicle {}.'.format(vehicle.id),
                        #           'Action {}.'.format(prediction + 1),
                        #           'Free Capacity {}.'.format(vehicle.free_capacity),
                        #           'User Ride Time {}.'.format(vehicle.ride_time),
                        #           'Next Free Time {}.'.format(vehicle.free_time),
                        #           'Coordinates {}.'.format(vehicle.coords))

                if num_finish == len(indices):
                    for user in users:
                        # Check if the user is served by the same vehicle.
                        if len(user.served_id) != 2 or user.served_id[0] != user.served_id[1]:
                            break_same.append(user.id)
                            print('* User {} is served by {}.'.format(user.id, user.served_id))

                        # Check if the request of the user is finished.
                        if user.status != 2:
                            break_done.append(user.id)
                            print('* The request of User {} is unfinished.'.format(user.id))

                    for vehicle in vehicles:
                        print('> Vehicle {}'.format(vehicle.id))
                        for index, node in enumerate(vehicle.route):
                            if 0 < node < 2 * num_users + 1:
                                vehicle.route[index] = nodes_to_users[node]

                        ground_truth = zip(vehicle.route[1:-1], vehicle.schedule[1:-1])
                        prediction = zip(vehicle.pred_route[1:-1], vehicle.pred_schedule[1:-1])
                        print('Ground truth:', [term[0] for term in ground_truth])
                        print('Prediction:', [term[0] for term in prediction])

                    eval_obj_true.append(objective)
                    eval_obj_pred.append(sum(vehicle.cost for vehicle in vehicles))
                    eval_window.append(len(break_window))
                    eval_ride_time.append(len(set(break_ride_time)))
                    eval_not_same.append(len(break_same))
                    eval_not_done.append(len(break_done))
                    eval_rela_gap.append(abs(eval_obj_true[-1] - eval_obj_pred[-1]) / eval_obj_true[-1] * 100)

                    print('> Objective')
                    print('Ground truth: {:.4f}'.format(eval_obj_true[-1]))
                    print('Prediction: {:.4f}'.format(eval_obj_pred[-1]))
                    print('Gap (%): {:.2f}'.format(eval_rela_gap[-1]))

                    print('> Constraint')
                    print('# Time Window: {}'.format(eval_window[-1]))
                    print('# Ride Time: {}\n'.format(eval_ride_time[-1]))

                    break

            if num_instance >= args.num_instances:
                break

        print('Cost (Rist 2021): {:.2f}'.format(eval_obj_true[0]))
        print('Cost (predicted): {:.2f}'.format(eval_obj_pred[0]))
        print('Diff. (%): {:.2f}'.format(eval_rela_gap[0]))
        print('# Time Window: {}'.format(eval_window[0]))
        print('# Ride Time: {}'.format(eval_ride_time[0]))

        print('Aver. Cost (Rist 2021): {:.2f}'.format(sum(eval_obj_true) / len(eval_obj_true)))
        print('Aver. Cost (predicted): {:.2f}'.format(sum(eval_obj_pred) / len(eval_obj_pred)))
        print('Aver. Diff. (%): {:.2f}'.format(sum(eval_rela_gap) / len(eval_rela_gap)))
        print('Aver. # Time Window: {:.2f}'.format(sum(eval_window) / len(eval_window)))
        print('Aver. # Ride Time: {:.2f}'.format(sum(eval_ride_time) / len(eval_ride_time)))

        print('# Not Same: {}'.format(np.sum(np.asarray(eval_not_same) > 0)))
        print('# Not Done: {}'.format(np.sum(np.asarray(eval_not_done) > 0)))

        path_result = './result/'
        os.makedirs(path_result, exist_ok=True)

        with open(path_result + 'evaluation.txt', 'a+') as output:
            json.dump({
                'Cost (Rist 2021)': round(eval_obj_true[0], 2),
                'Cost (predicted)': round(eval_obj_pred[0], 2),
                'Diff. (%)': round(eval_rela_gap[0], 2),
                '# Time Window': eval_window[0],
                '# Ride Time': eval_ride_time[0],
            }, output)
            output.write('\n')
            json.dump({
                'Aver. Cost (Rist 2021)': round(sum(eval_obj_true) / len(eval_obj_true), 2),
                'Aver. Cost (predicted)': round(sum(eval_obj_pred) / len(eval_obj_pred), 2),
                'Aver. Diff. (%)': round(sum(eval_rela_gap) / len(eval_rela_gap), 2),
                'Aver. # Time Window': round(sum(eval_window) / len(eval_window), 2),
                'Aver. # Ride Time': round(sum(eval_ride_time) / len(eval_ride_time), 2),
            }, output)
            output.write('\n')
            json.dump({
                '# Not Same': int(np.sum(np.asarray(eval_not_same) > 0)),
                '# Not Done': int(np.sum(np.asarray(eval_not_done) > 0)),
            }, output)
            output.write('\n')


if __name__ == '__main__':
    main()
