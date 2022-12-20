from environment import *


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

    parser.add_argument('--train_index', type=int, default=8)
    parser.add_argument('--num_subsets', type=int, default=3)
    parser.add_argument('--num_instances', type=int, default=500)
    parser.add_argument('--wait_time', type=int, default=7)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    darp = Darp(args, mode='supervise')

    path_dataset = './dataset/'
    os.makedirs(path_dataset, exist_ok=True)
    shutil.rmtree(path_dataset)
    print("Directory {} has been removed successfully".format(path_dataset))
    os.makedirs(path_dataset)

    data = []
    num_dataset = 1

    for num_instance in range(1, len(darp.list_instances) + 1):
        objective = darp.reset(num_instance - 1)

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
                action = darp.action(k)
                darp.step(k, action)

                data.append([state, action])

        print(num_dataset, num_instance, sys.getsizeof(data), len(data), objective)

        if num_instance % args.num_instances == 0:
            file = 'dataset-' + darp.name + '-' + str(num_dataset) + '.pt'
            print('Save {}.\n'.format(file))
            torch.save(data, path_dataset + file)
            data = []
            num_dataset += 1
            if num_dataset > args.num_subsets:
                break


if __name__ == '__main__':
    main()
