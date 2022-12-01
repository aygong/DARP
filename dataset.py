import os

import numpy as np
import json
import math
import sys
import argparse
import shutil

import torch

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

    parser.add_argument('--num_subsets', type=int, default=3)
    parser.add_argument('--num_instances', type=int, default=500)
    parser.add_argument('--index', type=int, default=8)
    parser.add_argument('--wait_time', type=int, default=5)

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


class Vehicle:
    def __init__(self):
        self.id = 0
        self.max_route_duration = 0
        self.max_capacity = 0
        self.route = []
        self.schedule = []
        self.ordinal = 0
        self.coords = []
        self.free_capacity = 0
        self.ride_time = {}
        self.free_time = 0.0
        self.duration = 0


def euclidean_distance(coord_start, coord_end):
    return math.sqrt((coord_start[0] - coord_end[0]) ** 2 + (coord_start[1] - coord_end[1]) ** 2)


def shift_window(time_window, time):
    return [max(0.0, time_window[0] - time), max(0.0, time_window[1] - time)]


def check_window(time_window, time):
    return time < time_window[0] or time > time_window[1]


def update_ride_time(vehicle, users, ride_time):
    for key in vehicle.ride_time:
        vehicle.ride_time[key] += ride_time
        users[key].ride_time += ride_time


def main():
    args = parse_arguments()
    instance_name = parameters[args.index][1] + str(parameters[args.index][2]) + '-' + str(parameters[args.index][3])
    path = './instance/' + instance_name + '-train.txt'

    nodes_to_users = {}
    for i in range(1, 2 * (parameters[args.index][3] + 1) - 1):
        if i <= parameters[args.index][3]:
            nodes_to_users[i] = i
        else:
            nodes_to_users[i] = i - parameters[args.index][3]

    path_dataset = './dataset/'
    os.makedirs(path_dataset, exist_ok=True)
    shutil.rmtree(path_dataset)
    print("Directory {} has been removed successfully".format(path_dataset))
    os.makedirs(path_dataset)

    data = []
    num_dataset = 1
    num_instance = 0

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
            for i in range(0, num_users):
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
                if user.id < num_users / 2:
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

            try:
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
            except IndexError:
                print("IndexError: list index out of range")
                continue

            # for n in range(0, num_vehicles):
            #     vehicle = vehicles[n]
            #     print('Vehicle {}.'.format(vehicle.id),
            #           'Max Route Duration {}.'.format(vehicle.max_route_duration),
            #           'Maximum Vehicle Capacity {}.'.format(vehicle.max_capacity),
            #           'Route {}.'.format(vehicle.route),
            #           'Schedule {}.'.format(vehicle.schedule),
            #           'Coordinates {}.'.format(vehicle.coords),
            #           'Free Capacity {}.'.format(vehicle.free_capacity))

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

                    # if vehicle.free_time + args.wait_time < vehicle.schedule[vehicle.ordinal]:
                    #     action = num_users + 1
                    # else:
                    #     if vehicle.route[vehicle.ordinal + 1] != 2 * num_users + 1:
                    #         node = vehicle.route[vehicle.ordinal + 1]
                    #         user = users[nodes_to_users[node] - 1]
                    #         action = user.id
                    #     else:
                    #         action = num_users

                    # if n == 1:
                    #     mask = [0 if user.flag == 2 else 1 for user in users] + [1, 1]
                    #     print(mask, action, mask[action])
                    # mask = [0 if user.flag == 2 else 1 for user in users] + [1, 1]
                    # if mask[action] != 1:
                    #     raise ValueError(mask, action, mask[action])

                    if vehicle.free_time + args.wait_time < vehicle.schedule[vehicle.ordinal]:
                        action = num_users + 1

                        vehicle.free_time += args.wait_time
                        update_ride_time(vehicle, users, args.wait_time)
                    else:
                        wait_time = vehicle.schedule[vehicle.ordinal] - vehicle.free_time
                        vehicle.free_time = vehicle.schedule[vehicle.ordinal]

                        if vehicle.route[vehicle.ordinal] != 0:
                            node = vehicle.route[vehicle.ordinal]
                            user = users[nodes_to_users[node] - 1]

                            update_ride_time(vehicle, users, wait_time)

                            if user.id not in vehicle.ride_time.keys():
                                if check_window(user.pickup_window, vehicle.free_time):
                                    raise ValueError(
                                        'The pick-up time window of User {} is broken: {:.2f} not in {}.'.format(
                                            user.id, vehicle.free_time, user.pickup_window))
                                vehicle.ride_time[user.id] = 0.0
                            else:
                                if user.ride_time - user.duration > max_ride_time + 1e-2:
                                    raise ValueError('The ride time of User {} is too long: {:.2f} > {:.2f}.'.format(
                                            user.id, user.ride_time - user.duration, max_ride_time))

                                if check_window(user.dropoff_window, vehicle.free_time):
                                    raise ValueError(
                                        'The drop-off time window of User {} is broken: {:.2f} not in {}.'.format(
                                            user.id, vehicle.free_time, user.dropoff_window))
                                del vehicle.ride_time[user.id]

                            vehicle.duration = user.duration
                            user.ride_time = 0.0

                        if vehicle.route[vehicle.ordinal + 1] < 2 * num_users + 1:
                            node = vehicle.route[vehicle.ordinal + 1]
                            user = users[nodes_to_users[node] - 1]
                            action = user.id

                            if user.id not in vehicle.ride_time.keys():
                                travel_time = euclidean_distance(vehicle.coords, user.pickup_coords)
                                window_start = user.pickup_window[0]
                                vehicle.coords = user.pickup_coords
                                vehicle.free_capacity -= user.load
                                user.served_by = vehicle.id
                                user.status = 1
                            else:
                                travel_time = euclidean_distance(vehicle.coords, user.dropoff_coords)
                                window_start = user.dropoff_window[0]
                                vehicle.coords = user.dropoff_coords
                                vehicle.free_capacity += user.load
                                user.served_by = num_vehicles
                                user.status = 2

                            if vehicle.free_time + vehicle.duration + travel_time > window_start + 1e-2:
                                ride_time = vehicle.duration + travel_time
                                vehicle.free_time += ride_time
                            else:
                                ride_time = window_start - vehicle.free_time
                                vehicle.free_time = window_start

                            update_ride_time(vehicle, users, ride_time)
                        else:
                            action = num_users

                            vehicle.coords = dest_depot_coords
                            vehicle.free_time = max_route_duration
                            vehicle.duration = 0

                        vehicle.ordinal += 1

                    # if n == 0:
                    #     print('Time {}.'.format(time),
                    #           'Vehicle {}.'.format(vehicle.id),
                    #           'Action {}.'.format(action),
                    #           'Free Capacity {}.'.format(vehicle.free_capacity),
                    #           'User Ride Time {}.'.format(vehicle.ride_time),
                    #           'Next Free Time {}.'.format(vehicle.free_time),
                    #           'Coordinates {}.'.format(vehicle.coords))

                    data.append([state, action])

                if num_finish == len(indices):
                    break

            if num_instance % args.num_instances == 0:
                file = 'dataset-' + instance_name + '-' + str(num_dataset) + '.pt'
                print('Save {}.\n'.format(file))
                torch.save(data, path_dataset + file)
                data = []
                num_dataset += 1
                if num_dataset > args.num_subsets:
                    break
            else:
                print(num_dataset, num_instance, sys.getsizeof(data), len(data), objective)


if __name__ == '__main__':
    main()
