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
    parser.add_argument('--index', type=int, default=9)
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
        self.pickup_coordinates = []
        self.dropoff_coordinates = []
        self.pickup_time_window = []
        self.dropoff_time_window = []
        self.service_duration = 0
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


class Vehicle:
    def __init__(self):
        self.id = 0
        self.max_route_duration = 0
        self.max_capacity = 0
        self.route = []
        self.schedule = []
        self.ordinal = 2
        self.coordinates = []
        self.free_capacity = 0
        self.user_ride_time = {}
        self.next_free_time = 0.0
        self.service_duration = 0
        self.route_pred = [0]
        self.schedule_pred = [0]
        self.cost = 0.0


def euclidean_distance(coord_start, coord_end):
    return math.sqrt((coord_start[0] - coord_end[0]) ** 2 + (coord_start[1] - coord_end[1]) ** 2)


def time_window_shift(time_window, time):
    return [max(time_window[0] - time, 0.0), max(time_window[1] - time, 0.0)]


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
        target_seq_len=num_users + 1,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        d_ff=args.d_ff,
        dropout=args.dropout)

    checkpoint = torch.load('./model/model-' + instance_name + '.model')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if cuda_available:
        model.cuda()

    num_instance = 0
    obj_true = []
    obj_pred = []
    list_time_window = []
    list_ride_time = []

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

            obj_true.append(objective)

            users = []
            for i in range(0, num_users):
                user = User()
                user.id = i
                user.max_ride_time = max_ride_time
                user.served_by = num_vehicles
                users.append(user)

            for i in range(0, 2 * (num_users + 1)):
                node = pair['instance'][i + 1]
                if i == 0:
                    origin_depot_coordinates = [float(node[1]), float(node[2])]
                    continue
                if i == 2 * (num_users + 1) - 1:
                    destination_depot_coordinates = [float(node[1]), float(node[2])]
                    continue
                user = users[nodes_to_users[i] - 1]
                if i <= num_users:
                    # Pick-up nodes
                    user.pickup_coordinates = [float(node[1]), float(node[2])]
                    user.service_duration = node[3]
                    user.load = node[4]
                    user.pickup_time_window = [float(node[5]), float(node[6])]
                else:
                    # Drop-off nodes
                    user.dropoff_coordinates = [float(node[1]), float(node[2])]
                    user.dropoff_time_window = [float(node[5]), float(node[6])]

            # for i in range(0, num_users):
            #     user = users[i]
            #     print('User {}'.format(user.id + 1),
            #           'Pick-up Coordinates {}.'.format(user.pickup_coordinates),
            #           'Drop-off Coordinates {}.'.format(user.dropoff_coordinates),
            #           'Pick-up Time Window {}.'.format(user.pickup_time_window),
            #           'Drop-off Time Window {}.'.format(user.dropoff_time_window),
            #           'Service Duration {}.'.format(user.service_duration),
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
                vehicle.coordinates = [0.0, 0.0]
                vehicle.free_capacity = max_vehicle_capacity
                vehicle.next_free_time = 0.0
                vehicles.append(vehicle)

            # for n in range(0, num_vehicles):
            #     vehicle = vehicles[n]
            #     print('Vehicle {}.'.format(vehicle.id),
            #           'Max Route Duration {}.'.format(vehicle.max_route_duration),
            #           'Maximum Vehicle Capacity {}.'.format(vehicle.max_capacity),
            #           'Route {}.'.format(vehicle.route),
            #           'Schedule {}.'.format(vehicle.schedule),
            #           'Coordinates {}.'.format(vehicle.coordinates),
            #           'Free Capacity {}.'.format(vehicle.free_capacity))

            num_time_window = 0
            num_ride_time = []

            while True:
                next_free_times = [vehicle.next_free_time for vehicle in vehicles]
                time = np.min(next_free_times)
                indices = np.argwhere(next_free_times == time)
                indices = indices.flatten().tolist()
                num_finish = 0

                for _, n in enumerate(indices):
                    vehicle = vehicles[n]
                    if vehicle.next_free_time == max_route_duration:
                        num_finish += 1
                        continue

                    for user in users:
                        if user.id in vehicle.user_ride_time.keys():
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
                                           [user.service_duration,
                                            user.load,
                                            user.status,
                                            user.served_by,
                                            user.ride_time,
                                            time_window_shift(user.pickup_time_window, time),
                                            time_window_shift(user.dropoff_time_window, time),
                                            vehicle.id,
                                            user.flag]
                                           + [vehicle.service_duration + euclidean_distance(
                                               vehicle.coordinates, user.pickup_coordinates)
                                              if user.status == 0 else
                                              vehicle.service_duration + euclidean_distance(
                                                  vehicle.coordinates, user.dropoff_coordinates)
                                              for vehicle in vehicles])) for user in users]

                    # Mask information.
                    # 0: waiting, 1: being served, 2: done
                    mask_info = [0 if user.flag == 2 else 1 for user in users]

                    state = [users_info, mask_info]

                    state, _ = DataLoader([state, 0], batch_size=1)  # noqa
                    outputs = model(state, device)
                    _, prediction = torch.max(f.softmax(outputs, dim=1), 1)

                    if prediction != num_users:
                        user = users[prediction]

                        if user.id not in vehicle.user_ride_time.keys():
                            travel_time = euclidean_distance(vehicle.coordinates, user.pickup_coordinates)
                            window_start = user.pickup_time_window[0]
                            vehicle.coordinates = user.pickup_coordinates
                            user.status = 1
                        else:
                            travel_time = euclidean_distance(vehicle.coordinates, user.dropoff_coordinates)
                            window_start = user.dropoff_time_window[0]
                            vehicle.coordinates = user.dropoff_coordinates
                            user.status = 2

                        vehicle.cost += travel_time

                        if vehicle.next_free_time + vehicle.service_duration + travel_time > window_start:
                            ride_time = vehicle.service_duration + travel_time
                            vehicle.next_free_time += ride_time
                        else:
                            ride_time = window_start - vehicle.next_free_time
                            vehicle.next_free_time = window_start

                        for key in vehicle.user_ride_time:
                            vehicle.user_ride_time[key] += ride_time
                            users[key].ride_time += ride_time
                            if users[key].ride_time - users[key].service_duration > max_ride_time + 1e-6:
                                if users[key].id >= num_users / 2 or users[key].id + 1 != vehicle.route_pred[-1]:
                                    print('The ride time of User {} is too long: {:.2f} > {:.2f}.'.format(
                                        users[key].id + 1, users[key].ride_time - users[key].service_duration, max_ride_time))
                                    num_ride_time.append(users[key].id + 1)

                        if user.id not in vehicle.user_ride_time.keys():
                            if vehicle.next_free_time < user.pickup_time_window[0] or \
                                    vehicle.next_free_time > user.pickup_time_window[1]:
                                print('The pick-up time window of User {} is broken: {:.2f} not in [{}, {}].'.format(
                                    user.id + 1, vehicle.next_free_time, user.pickup_time_window[0], user.pickup_time_window[1]))
                                num_time_window += 1
                            vehicle.user_ride_time[user.id] = 0.0
                            vehicle.free_capacity -= user.load
                            user.served_by = vehicle.id
                        else:
                            if vehicle.next_free_time < user.dropoff_time_window[0] or \
                                    vehicle.next_free_time > user.dropoff_time_window[1]:
                                print('The drop-off time window of User {} is broken: {:.2f} not in [{}, {}].'.format(
                                    user.id + 1, vehicle.next_free_time, user.dropoff_time_window[0], user.dropoff_time_window[1]))
                                num_time_window += 1
                            del vehicle.user_ride_time[user.id]
                            vehicle.free_capacity += user.load
                            user.served_by = num_vehicles

                        user.ride_time = 0.0
                        vehicle.service_duration = user.service_duration
                    else:
                        vehicle.cost += euclidean_distance(vehicle.coordinates, destination_depot_coordinates)
                        vehicle.next_free_time = max_route_duration
                        vehicle.coordinates = destination_depot_coordinates
                        vehicle.service_duration = 0

                    vehicle.route_pred.append(prediction.item() + 1)
                    vehicle.schedule_pred.append(vehicle.next_free_time)

                if num_finish == len(indices):
                    for vehicle in vehicles:
                        print('-> Vehicle {}'.format(vehicle.id))
                        for index, node in enumerate(vehicle.route):
                            if 0 < node < 2 * num_users + 1:
                                vehicle.route[index] = nodes_to_users[node]

                        ground_truth = zip(vehicle.route[1:-1], vehicle.schedule[1:-1])
                        prediction = zip(vehicle.route_pred[1:-1], vehicle.schedule_pred[1:-1])
                        print('Ground truth:', [term[0] for term in ground_truth])
                        print('Prediction:', [term[0] for term in prediction])
                        # print('Ground truth:', [f'({term[0]}, {term[1]:.2f})' for term in ground_truth])
                        # print('Prediction:', [f'({term[0]}, {term[1]:.2f})' for term in prediction])

                    obj_pred.append(sum(vehicle.cost for vehicle in vehicles))
                    list_time_window.append(num_time_window)
                    print(num_ride_time, len(set(num_ride_time)))
                    list_ride_time.append(len(set(num_ride_time)))

                    print('-> Objective')
                    print('Ground truth: {:.4f}.'.format(obj_true[-1]))
                    print('Prediction: {:.4f}.\n'.format(obj_pred[-1]))

                    break

            if num_instance >= args.num_instances:
                break

        print('Cost (Rist 2021): {:.2f}.'.format(obj_true[0]))
        print('Cost (predicted): {:.2f}.'.format(obj_pred[0]))
        print('Gap (%): {:.2f}.'.format((obj_true[0] - obj_pred[0]) / obj_true[0] * 100))
        print('# Time Window: {}.'.format(list_time_window[0]))
        print('# Ride Time: {}.'.format(list_ride_time[0]))

        aver_true = sum(obj_true) / len(obj_true)
        aver_pred = sum(obj_pred) / len(obj_pred)
        print('Aver. Cost (Rist 2021): {:.2f}.'.format(aver_true))
        print('Aver. Cost (predicted): {:.2f}.'.format(aver_pred))
        print('Gap (%): {:.2f}.'.format((aver_true - aver_pred) / aver_true * 100))
        print('Aver. # Time Window: {:.2f}.'.format(sum(list_time_window) / len(list_time_window)))
        print('Aver. # Ride Time: {:.2f}.'.format(sum(list_ride_time) / len(list_ride_time)))


if __name__ == '__main__':
    main()
