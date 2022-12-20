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


def euclidean_distance(coord_start, coord_end):
    return math.sqrt((coord_start[0] - coord_end[0]) ** 2 + (coord_start[1] - coord_end[1]) ** 2)


def shift_window(time_window, time):
    return [max(0.0, time_window[0] - time), max(0.0, time_window[1] - time)]


def check_window(time_window, time):
    return time < time_window[0] or time > time_window[1]


def update_ride_time(vehicle, users, ride_time):
    for uid in vehicle.serving:
        users[uid - 1].ride_time += ride_time


class User:
    def __init__(self, mode):
        super(User, self).__init__()

        self.id = 0
        self.pickup_coords = []
        self.dropoff_coords = []
        self.serve_duration = 0
        self.load = 0
        self.pickup_window = []
        self.dropoff_window = []
        self.ride_time = 0.0
        # alpha: status taking values in {0, 1, 2}
        # 0: the user is waiting to be served
        # 1: the user is being served by a vehicle
        # 2: the user has been served
        self.alpha = 0
        # beta: status taking values in {0, 1, 2}
        # 0: the user is waiting to be served
        # 1: the user is being served by the vehicle performing an action at time step t
        # 2: the user cannot be served by the vehicle
        self.beta = 0
        # Identity of the vehicle which is serving the user
        self.served = 0

        if mode == 'evaluate':
            self.served_id = []


class Vehicle:
    def __init__(self, mode):
        super(Vehicle, self).__init__()

        self.id = 0
        self.route = []
        self.schedule = []
        self.ordinal = 0
        self.coords = [0.0, 0.0]
        self.serving = []
        self.free_capacity = 0
        self.free_time = 0.0
        self.serve_duration = 0

        if mode == 'evaluate':
            self.pred_route = [0]
            self.pred_schedule = [0]
            self.cost = 0.0


class Darp:
    def __init__(self, args, mode, device=None):
        super(Darp, self).__init__()

        self.args = args
        self.mode = mode
        self.device = device

        self.train_type = parameters[args.train_index][1]
        self.train_K = parameters[args.train_index][2]
        self.train_N = parameters[args.train_index][3]
        self.train_T = parameters[args.train_index][4]
        self.train_Q = parameters[args.train_index][5]
        self.train_L = parameters[args.train_index][6]
        print('Instance Type: {}.'.format(self.train_type),
              'Number of vehicles: {}.'.format(self.train_K),
              'Number of users: {}.'.format(self.train_N),
              'Maximum route duration: {}.'.format(self.train_T),
              'Maximum vehicle capacity: {}.'.format(self.train_Q),
              'Maximum ride time: {}.'.format(self.train_L))

        self.node2user = {}
        for i in range(1, 2 * self.train_N + 1):
            if i <= self.train_N:
                self.node2user[i] = i
            else:
                self.node2user[i] = i - self.train_N

        self.name = self.train_type + str(self.train_K) + '-' + str(self.train_N)
        self.path = './instance/' + self.name + '-train' + '.txt'
        self.list_instances = []
        with open(self.path, 'r') as file:
            for instance in file:
                self.list_instances.append(json.loads(instance))

        self.users = []
        self.vehicles = []

    def reset(self, num_instance):
        instance = self.list_instances[num_instance]

        self.users = []
        self.vehicles = []

        for i in range(1, self.train_N + 1):
            user = User(mode=self.mode)
            user.id = i
            user.served = self.train_K
            self.users.append(user)

        for i in range(1, 2 * self.train_N + 1):
            node = instance['instance'][i + 1]  # noqa
            user = self.users[self.node2user[i] - 1]
            if i <= self.train_N:
                # Pick-up nodes
                user.pickup_coords = [float(node[1]), float(node[2])]
                user.serve_duration = node[3]
                user.load = node[4]
                user.pickup_window = [float(node[5]), float(node[6])]
            else:
                # Drop-off nodes
                user.dropoff_coords = [float(node[1]), float(node[2])]
                user.dropoff_window = [float(node[5]), float(node[6])]

        # Time-window tightening (Section 5.1.1, Cordeau 2006)
        for user in self.users:
            travel_time = euclidean_distance(user.pickup_coords, user.dropoff_coords)
            if user.id <= self.train_N / 2:
                # Drop-off requests
                user.pickup_window[0] = \
                    round(max(0.0, user.dropoff_window[0] - self.train_L - user.serve_duration), 3)
                user.pickup_window[1] = \
                    round(min(user.dropoff_window[1] - travel_time - user.serve_duration, self.train_T), 3)
            else:
                # Pick-up requests
                user.dropoff_window[0] = \
                    round(max(0.0, user.pickup_window[0] + user.serve_duration + travel_time), 3)
                user.dropoff_window[1] = \
                    round(min(user.pickup_window[1] + user.serve_duration + self.train_L, self.train_T), 3)

        for k in range(0, self.train_K):
            vehicle = Vehicle(mode=self.mode)
            vehicle.id = k
            vehicle.route = instance['routes'][k]  # noqa
            vehicle.route.insert(0, 0)
            vehicle.route.append(2 * self.train_N + 1)
            vehicle.schedule = instance['schedule'][k]  # noqa
            vehicle.free_capacity = self.train_Q
            self.vehicles.append(vehicle)

        return instance['objective']  # noqa

    def beta(self, k):
        for user in self.users:
            if user.alpha == 1 and user.served == self.vehicles[k].id:
                # 1: the user is being served by the vehicle performing an action at time step t
                user.beta = 1
            else:
                if user.alpha == 0:
                    if user.load <= self.vehicles[k].free_capacity:
                        # 0: the user is waiting to be served
                        user.beta = 0
                    else:
                        # 2: the user cannot be served by the vehicle
                        user.beta = 2
                else:
                    # 2: the user has been served
                    user.beta = 2

    def state(self, k, time):
        state = [list(map(np.float32,
                          [user.pickup_coords,
                           user.dropoff_coords,
                           shift_window(user.pickup_window, time),
                           shift_window(user.dropoff_window, time),
                           user.ride_time,
                           user.alpha,
                           user.beta,
                           user.served,
                           self.vehicles[k].id]
                          + [vehicle.serve_duration + euclidean_distance(
                              vehicle.coords, user.pickup_coords)
                             if user.alpha == 0 else
                             vehicle.serve_duration + euclidean_distance(
                                 vehicle.coords, user.dropoff_coords)
                             for vehicle in self.vehicles]))
                 for user in self.users]

        return state

    def action(self, k):
        vehicle = self.vehicles[k]
        r = vehicle.ordinal

        if vehicle.free_time + self.args.wait_time < vehicle.schedule[r]:
            # Wait at the present node
            action = self.train_N + 1
        else:
            if vehicle.route[r + 1] < 2 * self.train_N + 1:
                # Move to the next node
                node = vehicle.route[r + 1]
                action = self.node2user[node] - 1
            else:
                # Move to the destination depot
                action = self.train_N

        return action

    def step(self, k, action):
        vehicle = self.vehicles[k]
        r = vehicle.ordinal

        if vehicle.free_time + self.args.wait_time < vehicle.schedule[r]:
            # Wait at the present node
            vehicle.free_time += self.args.wait_time
            update_ride_time(vehicle, self.users, self.args.wait_time)
        else:
            wait_time = vehicle.schedule[r] - vehicle.free_time
            update_ride_time(vehicle, self.users, wait_time)
            vehicle.free_time = vehicle.schedule[r]

            if vehicle.route[r] != 0:
                # Start to serve the user at the present node
                node = vehicle.route[r]
                user = self.users[self.node2user[node] - 1]

                if user.id not in vehicle.serving:
                    # Check the pick-up time window
                    if check_window(user.pickup_window, vehicle.free_time):
                        raise ValueError('The pick-up time window of User {} is broken: {:.2f} not in {}.'.format(
                            user.id, vehicle.free_time, user.pickup_window))
                    # Append the user to the serving list
                    vehicle.serving.append(user.id)
                else:
                    # Check the ride time
                    if user.ride_time - user.serve_duration > self.train_L + 1e-2:
                        raise ValueError('The ride time of User {} is too long: {:.2f} > {:.2f}.'.format(
                            user.id, user.ride_time - user.serve_duration, self.train_L))
                    # Check the drop-off time window
                    if check_window(user.dropoff_window, vehicle.free_time):
                        raise ValueError('The drop-off time window of User {} is broken: {:.2f} not in {}.'.format(
                            user.id, vehicle.free_time, user.dropoff_window))
                    # Remove the user from the serving list
                    vehicle.serving.remove(user.id)

                vehicle.serve_duration = user.serve_duration
                user.ride_time = 0.0

            if vehicle.route[r + 1] < 2 * self.train_N + 1:
                # Move to the next node
                node = vehicle.route[r + 1]
                user = self.users[self.node2user[node] - 1]

                if user.id not in vehicle.serving:
                    travel_time = euclidean_distance(vehicle.coords, user.pickup_coords)
                    window_start = user.pickup_window[0]
                    vehicle.coords = user.pickup_coords
                    vehicle.free_capacity -= user.load
                    user.served = vehicle.id
                    user.alpha = 1
                else:
                    travel_time = euclidean_distance(vehicle.coords, user.dropoff_coords)
                    window_start = user.dropoff_window[0]
                    vehicle.coords = user.dropoff_coords
                    vehicle.free_capacity += user.load
                    user.served = self.train_K
                    user.alpha = 2

                if vehicle.free_time + vehicle.serve_duration + travel_time > window_start + 1e-2:
                    ride_time = vehicle.serve_duration + travel_time
                    vehicle.free_time += ride_time
                else:
                    ride_time = window_start - vehicle.free_time
                    vehicle.free_time = window_start

                update_ride_time(vehicle, self.users, ride_time)
            else:
                # Move to the destination depot
                vehicle.coords = [0.0, 0.0]
                vehicle.free_time = 1440
                vehicle.serve_duration = 0

            vehicle.ordinal += 1

    def finish(self):
        free_times = np.array([vehicle.free_time for vehicle in self.vehicles])
        num_finish = np.sum(free_times == 1440)

        if num_finish == self.train_K:
            flag = False
        else:
            flag = True

        return flag
