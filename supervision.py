from utils import *
from transformer import Transformer

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import random
import json

import torch
from torch import nn
from torch.utils.data import ConcatDataset, SubsetRandomSampler, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as f


def supervision(args):
    train_type, train_K, train_N, train_T, train_Q, train_L = load_instance(args.train_index, 'train')
    name = train_type + str(train_K) + '-' + str(train_N)
    path_dataset = ['./dataset/' + file for file in os.listdir('./dataset') if file.startswith('dataset-' + name)]
    training_sets = []
    for file in path_dataset:
        print('Load', file)
        training_sets.append(torch.load(file))
    training_set = ConcatDataset(training_sets)
    data_size = len(training_set)
    path_model = './model/'
    os.makedirs(path_model, exist_ok=True)

    path_result = './result/'
    os.makedirs(path_result, exist_ok=True)

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
    train_data = DataLoader(training_set, batch_size=args.batch_size, sampler=train_sampler)
    valid_data = DataLoader(training_set, batch_size=args.batch_size, sampler=valid_sampler)

    # Determine if your system supports CUDA
    cuda_available = torch.cuda.is_available()
    device = get_device(cuda_available)

    model = Transformer(
        device=device,
        num_vehicles=train_K,
        input_seq_len=train_N,
        target_seq_len=train_N + 2,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        d_ff=args.d_ff,
        dropout=args.dropout)

    model_name = name + '-' + str(args.wait_time)

    if cuda_available:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.99)

    epochs = args.epochs
    train_performance = np.zeros(epochs)
    valid_performance = np.zeros(epochs)
    exec_times = np.zeros(epochs)
    model_validation = True

    for epoch in range(epochs):
        print('--------Training for Epoch {} starting:--------'.format(epoch))
        start = time.time()
        running_loss = 0
        iters = 0
        model.train()

        for _, (states, actions) in enumerate(train_data):
            iters += 1
            actions = actions.to(device)

            optimizer.zero_grad()

            outputs = model(states)
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
            print('-------Validation for Epoch {} starting:-------'.format(epoch))
            model.eval()

            with torch.no_grad():
                valid_measure = [0, 0]

                # Loop over batches in an epoch using valid_data
                for _, (states, actions) in enumerate(valid_data):
                    actions = actions.to(device)

                    outputs = model(states)
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
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': criterion,
        }, './model/' + 'model-' + model_name + '.model')

        end = time.time()
        exec_time = end - start
        exec_times[epoch] = exec_time
        print('-> Execution time for Epoch {}: {:.4f} seconds.'.format(epoch, exec_time))
        print('-> Estimated execution time remaining: {:.4f} seconds.\n'.format(exec_time * (epochs - epoch - 1)))

        with open(path_result + 'training_log.txt', 'a+') as file:
            json.dump({
                'epoch': epoch,
                'execution time': exec_time / 3600,
                'estimated execution time remaining': exec_time * (epochs - epoch - 1) / 3600,
            }, file)
            file.write("\n")

    print('Training finished.')
    print('Average execution time per epoch: {:.4f} seconds.'.format(np.mean(exec_times)))
    print("Total execution time: {:.4f} seconds.".format(np.sum(exec_times)))  

    fig, ax = plt.subplots()
    file_name = 'accuracy-' + name + '-' + str(args.wait_time)
    ax.plot(np.arange(epochs), train_performance, label="Training accuracy")
    ax.plot(np.arange(epochs), valid_performance, label="Validation accuracy")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set(ylim=(0, 1))
    ax.legend()
    plt.savefig(path_result + file_name + '.pdf')

    with open(path_result + file_name + '.npy', 'wb') as file:
        np.save(file, train_performance)  # noqa
        np.save(file, valid_performance)  # noqa
