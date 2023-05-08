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

from graph_transformer import GraphTransformerNet


def supervision(args):
    train_type, train_K, train_N, train_T, train_Q, train_L = load_instance(args.train_index, 'train')
    name = train_type + str(train_K) + '-' + str(train_N)
    path_dataset = ['./dataset/' + file for file in os.listdir('./dataset') if file.startswith('dataset-' + name)]
    training_sets = []
    for file in path_dataset:
        print('Load', file)
        training_sets.append(torch.load(file))
        #os.remove(file)
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
    train_data = DataLoader(training_set, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate)
    valid_data = DataLoader(training_set, batch_size=args.batch_size, sampler=valid_sampler, collate_fn=collate)

    # Determine if your system supports CUDA
    cuda_available = torch.cuda.is_available()
    device = get_device(cuda_available)

    num_nodes = 2*train_N + train_K + 2
    model = GraphTransformerNet(
        device=device,
        num_nodes=num_nodes,
        num_node_feat=17,
        num_edge_feat=3,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        dropout=0.1
    )

    model_name = name + '-' + str(args.wait_time) +'-'+ str(args.filename_index)

    if cuda_available:
        model.cuda()

    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.99)

    epochs = args.epochs
    train_policy_performance = np.zeros(epochs)
    train_value_performance = np.zeros(epochs)
    valid_policy_performance = np.zeros(epochs)
    valid_value_performance = np.zeros(epochs)
    exec_times = np.zeros(epochs)
    model_validation = True

    for epoch in range(epochs):
        print('--------Training for Epoch {} starting:--------'.format(epoch))
        start = time.time()
        running_loss = 0
        iters = 0
        model.train()

        for _, (graphs, ks, action_nodes, values) in enumerate(train_data):
            if cuda_available:
                torch.cuda.empty_cache()

            iters += 1
            ks = ks.to(device)
            action_nodes = action_nodes.to(device)
            values = values.to(device, dtype=torch.float32)
            graphs = graphs.to(device)
            batch_x = graphs.ndata['feat'].to(device)
            batch_e = graphs.edata['feat'].to(device)

            optimizer.zero_grad()

            policy_outputs, value_outputs = model(graphs, batch_x, batch_e, ks, num_nodes, masking=True)
            policy_loss = criterion_policy(policy_outputs, action_nodes)
            value_loss = 0#criterion_value(value_outputs / values, torch.ones(values.size()).to(device))
            
            loss =  policy_loss + value_loss * args.loss_ratio
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            scheduler.step(running_loss)

            _, predicted = torch.max(f.softmax(policy_outputs, dim=1), 1)
            train_policy_measure = (predicted == action_nodes).sum().item() / action_nodes.size(0)
            train_value_measure = abs(value_outputs - values).sum().item() / values.size(0)

            if iters % 20 == 0:
                print('Epoch: {}.'.format(epoch),
                      'Iteration: {}.'.format(iters),
                      'Training policy loss: {:.4f}.'.format(policy_loss.item()),
                      'Training policy accuracy: {:.4f}.'.format(train_policy_measure)#,
                      #'Training value loss: {:.4f}.'.format(value_loss.item()),
                      #'Training value MAE: {:.4f}.'.format(train_value_measure)
                      )
            train_policy_performance[epoch] += train_policy_measure
            train_value_performance[epoch] += train_value_measure

        train_policy_performance[epoch] /= (iters + 1)
        train_value_performance[epoch] /= (iters + 1)

        if model_validation:
            # Validation
            print('-------Validation for Epoch {} starting:-------'.format(epoch))
            model.eval()

            with torch.no_grad():
                valid_measure = [0, 0, 0]

                # Loop over batches in an epoch using valid_data
                for _, (graphs, ks, action_nodes, values) in enumerate(valid_data):
                    ks = ks.to(device)
                    action_nodes = action_nodes.to(device)
                    values = values.to(device, dtype=torch.float32)
                    graphs = graphs.to(device)
                    batch_x = graphs.ndata['feat'].to(device)
                    batch_e = graphs.edata['feat'].to(device)

                    policy_outputs, value_outputs = model(graphs, batch_x, batch_e, ks, num_nodes, masking=True)
                    policy_loss = criterion_policy(policy_outputs, action_nodes)
                    value_loss = 0#criterion_value(value_outputs / values, torch.ones(values.size()).to(device))
                    loss =  policy_loss + value_loss * args.loss_ratio

                    _, predicted = torch.max(f.softmax(policy_outputs, dim=1), 1)
                    valid_measure[0] += (predicted == action_nodes).sum().item()
                    valid_measure[1] += abs(value_outputs - values).sum().item()
                    valid_measure[2] += action_nodes.size(0)

            valid_policy_performance[epoch] = valid_measure[0] / valid_measure[2]
            valid_value_performance[epoch] = valid_measure[1] / valid_measure[2]
            print('Validation policy accuracy: {:.4f}.'.format(valid_policy_performance[epoch]))
            #print('Validation value MAE: {:.4f}.'.format(valid_value_performance[epoch]))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'policy_loss': criterion_policy#,
            #'value_loss': criterion_value,
        }, './model/' + 'sl-' + model_name + '.model')

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
                'validation accuracy': valid_policy_performance[epoch],
            }, file)
            file.write("\n")

    print('Training finished.')
    print('Average execution time per epoch: {:.4f} seconds.'.format(np.mean(exec_times)))
    print("Total execution time: {:.4f} seconds.\n".format(np.sum(exec_times)))

    #fig, ax = plt.subplots()
    file_name = 'accuracy-' + name + '-' + str(args.wait_time) + '-' + str(args.filename_index)
    #ax.plot(np.arange(epochs), train_policy_performance, label="Training accuracy")
    #ax.plot(np.arange(epochs), valid_policy_performance, label="Validation accuracy")
    #ax.set_xlabel('Epoch')
    #ax.set(ylim=(0, 1))
    #ax.legend()
    #ax.set_ylabel('Accuracy')
    #plt.savefig(path_result + file_name + '.pdf')

    #with open(path_result + file_name + '.npy', 'wb') as file:
    #    np.save(file, train_policy_performance)  # noqa
    #    np.save(file, train_value_performance)  # noqa
    #    np.save(file, valid_policy_performance)  # noqa
    #    np.save(file, valid_value_performance)  # noqa

    np.savez(
        path_result + file_name + '.npz',
        train_policy_performance = train_policy_performance,
        train_value_performance = train_value_performance,
        valid_policy_performance = valid_policy_performance,
        valid_value_performance = valid_value_performance
        )
