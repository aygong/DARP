import numpy as np
from icecream import ic
import os
import argparse
import matplotlib.pyplot as plt
import time
import random

from torch.utils.data import ConcatDataset, SubsetRandomSampler, DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch
from torch import nn

from transformer import Transformer
import torch.nn.functional as f


parameters = [['0', 'a', 2, 16, 480, 3, 30],
              ['2', 'a', 2, 24, 720, 3, 30],
              ['6', 'a', 3, 36, 720, 3, 30],
              ['11', 'a', 4, 48, 720, 3, 30],
              ['14', 'a', 5, 60, 720, 3, 30],
              ['17', 'a', 6, 72, 720, 3, 30],
              ['20', 'a', 7, 84, 720, 3, 30],
              ['23', 'a', 8, 96, 720, 3, 30],
              ['24', 'b', 2, 16, 480, 6, 45],
              ['26', 'b', 2, 24, 720, 6, 45],
              ['30', 'b', 3, 36, 720, 6, 45],
              ['35', 'b', 4, 48, 720, 6, 45],
              ['38', 'b', 5, 60, 720, 6, 45],
              ['41', 'b', 6, 72, 720, 6, 45],
              ['44', 'b', 7, 84, 720, 6, 45],
              ['47', 'b', 8, 96, 720, 6, 45]]


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--index', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
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


def main():
    args = parse_arguments()
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

    path_dataset = ['./dataset/' + file for file in os.listdir('./dataset')]
    datasets = []
    for file in path_dataset:
        ic('Datafile folder:', file)
        datasets.append(torch.load(file))
    dataset = ConcatDataset(datasets)
    data_size = len(dataset)

    path_model = './model/'
    os.makedirs(path_model, exist_ok=True)

    path_plot = './plot/'
    os.makedirs(path_plot, exist_ok=True)

    torch.manual_seed(0)
    random.seed(0)

    indices = list(range(data_size))
    split = int(np.floor(0.02 * data_size))
    train_indices, valid_indices = indices[split:], indices[:split]
    print('The size of training set: {}.'.format(len(train_indices)),
          'The size of validation set: {}.\n'.format(len(valid_indices)))

    shuffle = True
    if shuffle:
        np.random.shuffle(train_indices)
        np.random.shuffle(valid_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    train_data = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    valid_data = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler)

    # Determine if your system supports CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print('CUDA is available. Utilize GPUs for computation.\n')
        device = torch.device("cuda")
    else:
        print('CUDA is not available. Utilize CPUs for computation.\n')
        device = torch.device("cpu")

    input_seq_len = num_users
    target_seq_len = num_users + 1

    model = Transformer(
        num_vehicles=num_vehicles,
        num_users=num_users,
        max_route_duration=max_route_duration,
        max_vehicle_capacity=max_vehicle_capacity,
        max_ride_time=max_ride_time,
        input_seq_len=input_seq_len,
        target_seq_len=target_seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        d_ff=args.d_ff,
        dropout=args.dropout)

    if cuda_available:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.99)

    epochs = args.epochs
    train_performance = np.zeros(epochs)
    valid_performance = np.zeros(epochs)
    model_validation = True

    for epoch in range(epochs):
        print('\033[1m--------Training for Epoch {} starting:--------\033[0m'.format(epoch))
        start = time.time()
        running_loss = 0
        iters = 0
        model.train()

        for _, (states, actions) in enumerate(train_data):
            iters += 1
            _, mask_info = states
            actions = actions.to(device)

            if args.mask == 'on':
                # print('Use mask in training.')
                mask = torch.zeros(len(actions), input_seq_len)
                for user_id in range(num_users):
                    for example_id in range(len(actions)):
                        mask[example_id][user_id - num_users] = mask_info[user_id][example_id]
                mask = mask.to(device)
            else:
                mask = None

            optimizer.zero_grad()

            outputs = model(states, device, mask)
            loss = criterion(outputs, actions)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            scheduler.step(running_loss)

            _, predicted = torch.max(f.softmax(outputs, dim=1), 1)
            train_measure = (predicted == actions).sum().item() / actions.size(0)
            if iters % 20 == 0:
                print('Epoch: {}.'.format(epoch),
                      'Iteration: {}.'.format(iters),
                      'Training loss: {:.4f}.'.format(loss.item()),
                      'Training accuracy: {:.4f}.'.format(train_measure)
                      )
            train_performance[epoch] = train_performance[epoch] + train_measure

        train_performance[epoch] /= (iters + 1)

        if model_validation:
            # Validation
            print('\033[1m-------Validation for Epoch {} starting:-------\033[0m'.format(epoch))
            model.eval()

            with torch.no_grad():
                valid_measure = [0, 0]

                # Loop over batches in an epoch using valid_data
                for _, (states, actions) in enumerate(valid_data):
                    _, mask_info = states
                    actions = actions.to(device)

                    if args.mask == 'on':
                        # print('Use mask in validation.')
                        mask = torch.zeros(len(actions), input_seq_len)
                        for user_id in range(num_users):
                            for example_id in range(len(actions)):
                                mask[example_id][user_id - num_users] = mask_info[user_id][example_id]
                        mask = mask.to(device)
                    else:
                        mask = None

                    outputs = model(states, device, mask)
                    loss = criterion(outputs, actions)

                    _, predicted = torch.max(f.softmax(outputs, dim=1), 1)
                    valid_measure[0] += (predicted == actions).sum().item()
                    valid_measure[1] += actions.size(0)

            valid_performance[epoch] = valid_measure[0] / valid_measure[1]
            print('Validation accuracy: {:.4f}.'.format(valid_performance[epoch]))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, './model/' + 'model.model')

        end = time.time()
        exec_time = end - start
        print('-> Execution time for Epoch {}: {:.4f} seconds.'.format(epoch, exec_time))
        print('-> Estimated execution time remaining: {:.4f} seconds.\n'.format(exec_time * (epochs - epoch - 1)))

    fig, ax = plt.subplots()
    file_name = 'accuracy-' + str(num_vehicles) + '-' + str(num_users)
    ax.plot(np.arange(epochs), train_performance, label="Training accuracy")
    ax.plot(np.arange(epochs), valid_performance, label="Validation accuracy")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    # ax.set(ylim=(0, 1))
    ax.legend()
    plt.savefig(path_plot + file_name + '.pdf')

    with open(path_plot + file_name + '.npy', 'wb') as file:
        np.save(file, train_performance)  # noqa
        np.save(file, valid_performance)  # noqa


if __name__ == '__main__':
    main()
