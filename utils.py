import math
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


def load_instance(index, mode):
    _type_, K, N, T, Q, L = parameters[index][1:]
    if mode == 'train':
        print('Training instances -> Type: {}.'.format(_type_), 'K: {}.'.format(K), 'N: {}.'.format(N),
              'T: {}.'.format(T), 'Q: {}.'.format(Q), 'L: {}.'.format(L), '\n')
    else:
        print('Test instances -> Type: {}.'.format(_type_), 'K: {}.'.format(K), 'N: {}.'.format(N),
              'T: {}.'.format(T), 'Q: {}.'.format(Q), 'L: {}.'.format(L), '\n')

    return _type_, K, N, T, Q, L

def get_device(cuda_available):
    if cuda_available:
        print('CUDA is available. Utilize GPUs for computation.\n')
        device = torch.device("cuda")
    else:
        print('CUDA is not available. Utilize CPUs for computation.\n')
        device = torch.device("cpu")
    return device

def get_device(cuda_available):
    if cuda_available:
        print('CUDA is available. Utilize GPUs for computation.\n')
        device = torch.device("cuda")
    else:
        print('CUDA is not available. Utilize CPUs for computation.\n')
        device = torch.device("cpu")
    return device


def node_to_user(N):
    node2user = {}
    for i in range(1, 2 * N + 1):
        if i <= N:
            node2user[i] = i
        else:
            node2user[i] = i - N

    return node2user
