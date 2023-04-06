from utils import *

import numpy as np
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as f

import dgl


class User:
    def __init__(self, mode):
        super(User, self).__init__()

        self.id = 0
        self.pickup_coords = [0.0, 0.0]
        self.dropoff_coords = [0.0, 0.0]
        self.serve_duration = 0
        self.load = 0
        self.pickup_window = [0, 0]
        self.dropoff_window = [0, 0]
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

        if mode != 'supervise':
            self.pred_served = []


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

        if mode != 'supervise':
            self.pred_route = [0]
            self.pred_schedule = [0]
            self.pred_cost = 0.0


class Darp:
    def __init__(self, args, mode, device=None):
        super(Darp, self).__init__()

        self.args = args
        self.mode = mode
        self.device = device
        self.model = None
        self.logs = True
        self.log_probs = None

        # Load the parameters of training instances
        self.train_type, self.train_K, self.train_N, self.train_T, self.train_Q, self.train_L = \
            load_instance(args.train_index, 'train')
        # Set the name of training instances
        self.train_name = self.train_type + str(self.train_K) + '-' + str(self.train_N)

        if self.mode != 'evaluate':
            # Get the node-user mapping dictionary of training instances
            self.node2user = node_to_user(self.train_N)
            # Set the path of training instances
            self.data_path = './instance/' + self.train_name + '-train' + '.txt'
        else:
            # Load the parameters of test instances
            self.test_type, self.test_K, self.test_N, self.test_T, self.test_Q, self.test_L = \
                load_instance(args.test_index, 'test')
            # Get the node-user mapping dictionary of test instances
            self.node2user = node_to_user(self.test_N)
            # Set the name of test instances
            self.test_name = self.test_type + str(self.test_K) + '-' + str(self.test_N)
            # Set the path of test instances
            self.data_path = './instance/' + self.test_name + '-test' + '.txt'

        # Load instances
        self.list_instances = []
        self.load_from_file()

        # Initialize the lists of vehicles and users
        self.users = []
        self.vehicles = []

        if self.mode != 'supervise':
            # Initialize the lists of metrics
            self.break_window = []
            self.break_ride_time = []
            self.break_same = []
            self.break_done = []
            self.time_penalty = 0
            self.indices = []  # for beam search
            self.time = 0.0

    def load_from_file(self, num_instance=None):
        """ Load the instances from the file, in beam search we load the instances one by one """
        if num_instance:
            instance = self.list_instances[num_instance]
            self.list_instances = [instance]
        else:
            with open(self.data_path, 'r') as file:
                for instance in file:
                    self.list_instances.append(json.loads(instance))

    def reset(self, num_instance):
        K, N, T, Q, L = self.parameter()
        instance = self.list_instances[num_instance]

        self.users = []
        for i in range(1, N + 1):
            user = User(mode=self.mode)
            user.id = i
            user.served = self.train_K
            self.users.append(user)
        # Add dummy users
        for _ in range(0, self.train_N - N):
            user = User(mode=self.mode)
            user.alpha = 2
            user.beta = 2
            user.served = self.train_K
            self.users.append(user)

        for i in range(1, 2 * N + 1):
            node = instance['instance'][i + 1]  # noqa
            user = self.users[self.node2user[i] - 1]
            if i <= N:
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
            if user.id <= N / 2:
                # Drop-off requests
                user.pickup_window[0] = \
                    round(max(0.0, user.dropoff_window[0] - L - user.serve_duration), 3)
                user.pickup_window[1] = \
                    round(min(user.dropoff_window[1] - travel_time - user.serve_duration, T), 3)
            else:
                # Pick-up requests
                user.dropoff_window[0] = \
                    round(max(0.0, user.pickup_window[0] + user.serve_duration + travel_time), 3)
                user.dropoff_window[1] = \
                    round(min(user.pickup_window[1] + user.serve_duration + L, T), 3)

        self.vehicles = []
        for k in range(0, K):
            vehicle = Vehicle(mode=self.mode)
            vehicle.id = k
            vehicle.route = instance['routes'][k]  # noqa
            vehicle.route.insert(0, 0)
            vehicle.route.append(2 * N + 1)
            vehicle.schedule = instance['schedule'][k]  # noqa
            vehicle.free_capacity = Q
            self.vehicles.append(vehicle)
        # Add dummy vehicles
        for _ in range(0, self.train_K - K):
            vehicle = Vehicle(mode=self.mode)
            vehicle.free_time = 1440
            self.vehicles.append(vehicle)

        if self.mode != 'supervise':
            # Reinitialize the lists of metrics
            self.break_window = []
            self.break_ride_time = []
            self.break_same = []
            self.break_done = []
            self.time_penalty = 0

        return instance['objective']  # noqa

    def beta(self, k):
        _, N, _, _, _ = self.parameter()

        for i in range(0, N):
            user = self.users[i]
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
    
    def vehicle_present(self, u_coords):
        for k in self.vehicles:
            if k.coords == u_coords:
                # Must verify that there is not twice the same coordinates for two different users
                return k
        return None
    
    def state_graph(self, k, time):
        """
        Construct a graph of the situation
        Nodes: 
        - 2 per user (pickup + dropoff)
        - 1 source station per vehicle 
        - 1 destination station
        - 1 for the waiting action
        - total = 2N+K+2.
        Edges:
        - each vehicle is connected to available user locations and destination station if the vehicle is empty.
        - destination is connected to drop-off locations.
        - already visited nodes are not connected to anything.
        - the rest of the locations (users) are connected all together.
        """
        _, _, T, _, L = self.parameter()
        K, N = self.train_K, self.train_N
        n_nodes = 2*N + K + 2
        n_features = 17
        node_features = torch.zeros(n_nodes, n_features) # input features of each node
        node_info = [] # node info to draw edges: node number, user (if any), vehicle (if any), type (pickup, dropoff, wait, source, destination), is_next_available (true or false), coords
        next_vehicle_node = -1 # node conatining the vehicle that will perform an action
        for u in self.users:
            # pickup node
            window = shift_window(u.pickup_window, time)
            node_features[u.id, one_hot_node_type('pickup')] = 1    # Type of node (pickup, dropoff, wait, source, destination)
            node_features[u.id, 5] = u.pickup_coords[0]             # x coord
            node_features[u.id, 6] = u.pickup_coords[1]             # y coord
            node_features[u.id, 7] = window[0]                      # start of window
            node_features[u.id, 8] = window[1]                      # end of window
            node_features[u.id, 9] = u.serve_duration               # service time
            node_features[u.id, 10] = L - u.ride_time               # remaining ride time
            node_features[u.id, 11] = u.load                        # user load
            k_pres = self.vehicle_present(u.pickup_coords)          # check whether the is a vehicle on that node
            if k_pres:
                node_features[u.id, 12] = 1                         # vehicle present
                node_features[u.id, 13] = k_pres.free_capacity      # free capacity
                node_features[u.id, 14] = k_pres.free_time          # next available time
                node_features[u.id, 15] = T                         # Remaining route duration ??? TO CHANGE
                if k == k_pres.id:
                    node_features[u.id, 16] = 1                     # is next available
                    next_vehicle_node = u.id
            
            node_info.append((u.id, u, k_pres, 'pickup', (k_pres and k_pres.id==k), u.pickup_coords))

            # dropoff node
            window = shift_window(u.dropoff_window, time)
            node_features[u.id+N, one_hot_node_type('dropoff')] = 1   # Type of node (pickup, dropoff, wait, source, destination)
            node_features[u.id+N, 5] = u.dropoff_coords[0]            # x coord
            node_features[u.id+N, 6] = u.dropoff_coords[1]            # y coord
            node_features[u.id+N, 7] = window[0]                      # start of window
            node_features[u.id+N, 8] = window[1]                      # end of window
            node_features[u.id+N, 9] = u.serve_duration               # service time
            node_features[u.id+N, 10] = L - u.ride_time               # remaining ride time
            node_features[u.id+N, 11] = -u.load                       # - user load
            k_pres = self.vehicle_present(u.dropoff_coords)           # check whether the is a vehicle on that node
            if k_pres:
                node_features[u.id+N, 12] = 1                         # vehicle present
                node_features[u.id+N, 13] = k_pres.free_capacity      # free capacity
                node_features[u.id+N, 14] = k_pres.free_time          # next available time
                node_features[u.id+N, 15] = T                         # Remaining route duration ??? TO CHANGE
                if k == k_pres.id:
                    node_features[u.id+N, 16] = 1                     # is next available
                    next_vehicle_node = u.id+N

            node_info.append((u.id+N, u, k_pres, 'dropoff', (k_pres and k_pres.id==k), u.dropoff_coords))

        # Destination node
        node_features[2*N + 1, one_hot_node_type('destination')] = 1
        node_info.append((2*N+1, None, None, 'destination', False, [0.0, 0.0]))

        # Waiting node
        node_features[0, one_hot_node_type('wait')] = 1
        node_features[0, 5] = self.vehicles[k].coords[0]
        node_features[0, 6] = self.vehicles[k].coords[1]
        node_info.append((0, None, None, 'wait', False, self.vehicles[k].coords))

        # Source nodes, one for each vehicle
        for k_v in self.vehicles:
            node_features[2*N + 2 + k_v.id, one_hot_node_type('source')] = 1
            v_pres = None
            if k_v.coords == [0.0, 0.0] and k_v.free_time < 1440: # if vehicle still at source station
                v_pres = k_v
                node_features[2*N + 2 + k_v.id, 10] = 1
                node_features[2*N + 2 + k_v.id, 11] = k_v.free_capacity
                node_features[2*N + 2 + k_v.id, 12] = k_v.free_time
                node_features[2*N + 2 + k_v.id, 13] = T
                if k == k_v.id:
                    node_features[2*N + 2 + k_v.id, 14] = 1
                    next_vehicle_node = 2*N + 2 + k_v.id

            node_info.append((2*N+2+k_v.id, None, v_pres, 'source', k_v.id == k, [0.0, 0.0]))

        # Create a DGL Graph
        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        g.ndata['feat'] = node_features

        # edges
        edges = {
            'src':[],
            'dst':[],
            'feat':[]
        }
        for i, (i_u, u, k_u, t_u, u_next, u_coords) in enumerate(node_info):
            for (i_v, v, k_v, t_v, v_next, v_coords) in node_info[i+1:]:
                if is_edge(u, k_u, t_u, u_next, v, k_v, t_v, v_next):
                    pairing = 1 if (u and u==v) else 0
                    waiting = 1 if (t_u=='wait' or t_v=='wait') else 0
                    edge_feat = torch.tensor([euclidean_distance(u_coords, v_coords), pairing, waiting])
                    #g.add_edges(i_u, i_v, data={'feat':edge_feat})
                    edges['src'].append(i_u)
                    edges['dst'].append(i_v)
                    edges['feat'].append(edge_feat)

        g.add_edges(edges['src'], edges['dst'], data={'feat':torch.stack(edges['feat'])})
        
        g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
        if next_vehicle_node == -1:
            raise ValueError('No node found for next vehicle')
        return g, next_vehicle_node
    
    def action2node(self, action):
        """
        given an action, returns the corresponding node in the sate graph
        """
        if action < self.train_N: # user
            if self.users[action].alpha == 0:# and not self.vehicle_present(self.users[action].pickup_coords):
                return action+1 # pickup
            else:
                return (action+1) + self.train_N # dropoff
        elif action == self.train_N: # destination
            return 2*action + 1 # 2N + 1
        elif action == self.train_N + 1: # wait
            return torch.tensor(0, device= self.device) 
        else:
            raise RuntimeError('Action not recognized')
    
    def node2action(self, node):
        """
        given an node in the state graph, returns the corresponding action
        """
        if node == 0: # wait
            return torch.tensor(self.train_N + 1, device= self.device)
        elif node <= 2*self.train_N: # user
            return ((node-1) % self.train_N)
        elif node == 2*self.train_N + 1: # destination
            return torch.tensor(self.train_N, device= self.device)
        else:
            raise RuntimeError('Action not recognized')
    
    
    # noinspection PyMethodMayBeStatic
    def will_pick_up(self, vehicle, user):
        travel_time = euclidean_distance(vehicle.coords, user.pickup_coords)
        window_start = user.pickup_window[0]
        vehicle.coords = user.pickup_coords
        vehicle.free_capacity -= user.load
        user.served = vehicle.id
        user.alpha = 1

        return travel_time, window_start

    def will_drop_off(self, vehicle, user):
        travel_time = euclidean_distance(vehicle.coords, user.dropoff_coords)
        window_start = user.dropoff_window[0]
        vehicle.coords = user.dropoff_coords
        vehicle.free_capacity += user.load
        user.served = self.train_K
        user.alpha = 2

        return travel_time, window_start

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

    def supervise_step(self, k):
        """
        Simulate a step of the instance to reach the next state.
        Return the travel time of the transition.
        """

        vehicle = self.vehicles[k]
        r = vehicle.ordinal

        if vehicle.free_time + self.args.wait_time < vehicle.schedule[r]:
            # Wait at the present node
            vehicle.free_time += self.args.wait_time
            update_ride_time(vehicle, self.users, self.args.wait_time)
            travel_time = 0.0
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
                    travel_time, window_start = self.will_pick_up(vehicle, user)
                else:
                    travel_time, window_start = self.will_drop_off(vehicle, user)

                if vehicle.free_time + vehicle.serve_duration + travel_time > window_start + 1e-2:
                    ride_time = vehicle.serve_duration + travel_time
                    vehicle.free_time += ride_time
                else:
                    ride_time = window_start - vehicle.free_time
                    vehicle.free_time = window_start

                update_ride_time(vehicle, self.users, ride_time)
            else:
                # Move to the destination depot
                travel_time = euclidean_distance(vehicle.coords, [0.0, 0.0])
                vehicle.coords = [0.0, 0.0]
                vehicle.free_time = 1440
                vehicle.serve_duration = 0

            vehicle.ordinal += 1

        return travel_time

    def predict(self, graph, vehicle_node_id, user_mask=None, src_mask=None):
        #graph, ks, _, _ = DataLoader([graph, vehicle_node_id, 0, 0], batch_size=1, collate_fn=collate)  # noqa
        graph = graph.to(self.device)
        ks = torch.tensor([vehicle_node_id], device=self.device)
        batch_x = graph.ndata['feat'].to(self.device)
        batch_e = graph.edata['feat'].to(self.device)

        if self.mode == 'evaluate':
            #pred_mask = [0 if self.users[i].beta == 2 else 1 for i in range(0, self.test_N)] + \
            #            [0 for _ in range(0, self.train_N - self.test_N)] + [1, 1]
            #pred_mask = torch.Tensor(pred_mask).to(self.device)
            policy_outputs, value_outputs = self.model(graph, batch_x, batch_e, ks, masking=True)
            #policy_outputs = policy_outputs.masked_fill(pred_mask == 0, -1e6)
        else:
            policy_outputs, value_outputs = self.model(graph, batch_x, batch_e, ks, masking=True)

        probs = f.softmax(policy_outputs, dim=1)
        _, action_node = torch.max(probs, 1)

        # value outputs to also be returned in the future
        return action_node.item(), probs

    def evaluate_step(self, k, action):
        K, N, T, Q, L = self.parameter()
        vehicle = self.vehicles[k]

        if action == self.train_N + 1:
            # Wait at the present node
            vehicle.free_time += self.args.wait_time
            update_ride_time(vehicle, self.users, self.args.wait_time)
        else:
            if vehicle.pred_route[-1] != 0:
                # Start to serve the user at the present node
                user = self.users[vehicle.pred_route[-1] - 1]

                if user.id not in vehicle.serving:
                    # Check the pick-up time window
                    if check_window(user.pickup_window, vehicle.free_time) and user.id > N / 2:
                        if self.logs:
                            print('The pick-up time window of User {} is broken: {:.2f} not in {}.'.format(
                                user.id, vehicle.free_time, user.pickup_window))
                        self.break_window.append(user.id)
                        self.time_penalty += vehicle.free_time - user.pickup_window[0]
                    # Append the user to the serving list
                    vehicle.serving.append(user.id)
                else:
                    # Check the ride time
                    if user.ride_time - user.serve_duration > L + 1e-2:
                        if self.logs:
                            print('The ride time of User {} is too long: {:.2f} > {:.2f}.'.format(
                                user.id, user.ride_time - user.serve_duration, L))
                        self.break_ride_time.append(user.id)
                        self.time_penalty += user.ride_time - user.serve_duration - L
                    # Check the drop-off time window
                    if check_window(user.dropoff_window, vehicle.free_time) and user.id <= N / 2:
                        if self.logs:
                            print('The drop-off time window of User {} is broken: {:.2f} not in {}.'.format(
                                user.id, vehicle.free_time, user.dropoff_window))
                        self.break_window.append(user.id)
                        self.time_penalty += vehicle.free_time - user.dropoff_window[0]
                    # Remove the user from the serving list
                    vehicle.serving.remove(user.id)

                vehicle.serve_duration = user.serve_duration
                user.ride_time = 0.0

            if action < N:
                # Move to the next node
                user = self.users[action]

                if user.id not in vehicle.serving:
                    travel_time, window_start = self.will_pick_up(vehicle, user)
                    user.pred_served.append(vehicle.id)
                else:
                    travel_time, window_start = self.will_drop_off(vehicle, user)
                    user.pred_served.append(vehicle.id)

                if vehicle.free_time + vehicle.serve_duration + travel_time > window_start + 1e-2:
                    ride_time = vehicle.serve_duration + travel_time
                    vehicle.free_time += ride_time
                else:
                    ride_time = window_start - vehicle.free_time
                    vehicle.free_time = window_start

                vehicle.pred_cost += travel_time
                update_ride_time(vehicle, self.users, ride_time)
            else:
                # Move to the destination depot
                vehicle.pred_cost += euclidean_distance(vehicle.coords, [0.0, 0.0])
                vehicle.coords = [0.0, 0.0]
                vehicle.free_time = 1440
                vehicle.serve_duration = 0

            vehicle.pred_route.append(action + 1)
            vehicle.pred_schedule.append(vehicle.free_time)

    def finish(self):
        free_times = np.array([vehicle.free_time for vehicle in self.vehicles])
        num_finish = np.sum(free_times == 1440)

        if num_finish == self.train_K:
            flag = False
            if self.mode != 'supervise':
                _, N, _, _, _ = self.parameter()

                for i in range(0, N):
                    user = self.users[i]
                    # Check if the user is served by the same vehicle.
                    if len(user.pred_served) != 2 or user.pred_served[0] != user.pred_served[1]:
                        self.break_same.append(user.id)
                        print('* User {} is served by {}.'.format(user.id, user.pred_served))
                    # Check if the request of the user is finished.
                    if user.alpha != 2:
                        self.break_done.append(user.id)
                        print('* The request of User {} is unfinished.'.format(user.id))
        else:
            flag = True

        return flag

    def cost(self):
        return sum(vehicle.pred_cost for vehicle in self.vehicles)

    def parameter(self):
        if self.mode != 'evaluate':
            return self.train_K, self.train_N, self.train_T, self.train_Q, self.train_L
        else:
            return self.test_K, self.test_N, self.test_T, self.test_Q, self.test_L
    