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


def euclidean_distance(coord_start, coord_end):
    return math.sqrt((coord_start[0] - coord_end[0]) ** 2 + (coord_start[1] - coord_end[1]) ** 2)


def time_window_shift(time_window, time):
    return [max(time_window[0] - time, 0.0), max(time_window[1] - time, 0.0)]


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
            #     print('User {}'.format(user.id),
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

                    ordinal = vehicle.ordinal - 1
                    if vehicle.route[ordinal] != 2 * num_users + 1:
                        node = vehicle.route[ordinal]
                        user = users[nodes_to_users[node] - 1]
                        action = user.id

                        if user.id not in vehicle.user_ride_time.keys():
                            travel_time = euclidean_distance(vehicle.coordinates, user.pickup_coordinates)
                            vehicle.coordinates = user.pickup_coordinates
                            user.status = 1
                        else:
                            travel_time = euclidean_distance(vehicle.coordinates, user.dropoff_coordinates)
                            vehicle.coordinates = user.dropoff_coordinates
                            user.status = 2

                        ride_time = vehicle.schedule[ordinal] - vehicle.next_free_time
                        if vehicle.route[ordinal - 1] != 0:
                            if travel_time + vehicle.service_duration > ride_time + 1e-2:
                                print(travel_time + vehicle.service_duration, ride_time)
                                raise ValueError('The departure time is too early.')

                        for key in vehicle.user_ride_time:
                            vehicle.user_ride_time[key] += ride_time
                            users[key].ride_time += ride_time
                            if users[key].ride_time - users[key].service_duration > max_ride_time + 1e-6:
                                print(users[key].ride_time - users[key].service_duration, max_ride_time)
                                raise ValueError('The ride time of User {} is too long.'.format(users[key].id))

                        vehicle.next_free_time = vehicle.schedule[ordinal]

                        if user.id not in vehicle.user_ride_time.keys():
                            if vehicle.next_free_time < user.pickup_time_window[0] or \
                                    vehicle.next_free_time > user.pickup_time_window[1]:
                                raise ValueError('The pick-up time window of User {} is broken.'.format(user.id))
                            vehicle.user_ride_time[user.id] = 0.0
                            vehicle.free_capacity -= user.load
                            user.served_by = vehicle.id
                        else:
                            if vehicle.next_free_time < user.dropoff_time_window[0] or \
                                    vehicle.next_free_time > user.dropoff_time_window[1]:
                                raise ValueError('The drop-off time window of User {} is broken.'.format(user.id))
                            del vehicle.user_ride_time[user.id]
                            vehicle.free_capacity += user.load
                            user.served_by = num_vehicles

                        user.ride_time = 0.0
                        vehicle.service_duration = user.service_duration
                    else:
                        action = num_users
                        vehicle.next_free_time = max_route_duration
                        vehicle.coordinates = destination_depot_coordinates
                        vehicle.service_duration = 0

                    vehicle.ordinal += 1

                    # if n == 1:
                    #     print('Time {}.'.format(time),
                    #           'Vehicle {}.'.format(vehicle.id),
                    #           'Action {}.'.format(action),
                    #           'Free Capacity {}.'.format(vehicle.free_capacity),
                    #           'User Ride Time {}.'.format(vehicle.user_ride_time),
                    #           'Next Free Time {}.'.format(vehicle.next_free_time),
                    #           'Coordinates {}.'.format(vehicle.coordinates))

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
