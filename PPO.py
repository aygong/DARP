import random
import torch
import torch.nn.functional as f
from torch.distributions.categorical import Categorical
from environment import *
from utils import *

class PPO:
    def __init__(
            self,
            args,
            device,
            actor_critic_model,
            n_actions,
            gamma,
            lr,
            clip_rate,
            batch_size,
            n_epochs,
            collect_episodes,
            update_epochs,
            num_instances,
            constraint_penalty_alpha = 10.0
            ):
        
        self.args = args
        self.device = device
        self.model = actor_critic_model
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.clip_rate = clip_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.collect_episodes = collect_episodes
        self.update_epochs = update_epochs
        self.num_instances = num_instances
        self.constraint_penalty_alpha = constraint_penalty_alpha
        
        # Episodes data
        self.states = []
        self.action_nodes = []
        self.costs = []
        self.log_probs = []
        self.dones = []
        self.values = []
        self.vehicle_node_ids = []

        self.darp = Darp(args, mode='reinforce', device=device)

    def greedy_action(self, graph, vehicle_node_id):
        # Select action with max probability
        graph = graph.to(self.device)
        k = torch.tensor([vehicle_node_id], device=self.device)
        x = graph.ndata['feat'].to(self.device)
        e = graph.edata['feat'].to(self.device)

        with torch.no_grad():
            policy, value = self.model(graph, x, e, k, masking=True)
            probs = f.softmax(policy, dim=1)
            a = torch.argmax(probs).item()
            p = probs[a].item()
        return a, p, value
    
    def select_action(self, graph, vehicle_node_id):
        # Stochastic action selection for one state
        graph = graph.to(self.device)
        k = torch.tensor([vehicle_node_id], device=self.device)
        x = graph.ndata['feat'].to(self.device)
        e = graph.edata['feat'].to(self.device)

        with torch.no_grad():
            policy, value = self.model(graph, x, e, k, masking=True)
            logits = f.softmax(policy, dim=1)
            probs = Categorical(logits=logits)

            action = probs.sample()
            
        return action, probs.log_prob(action), value
    
    def evaluate_policy(self, states, vehicle_node_ids, action_nodes):
        # Policy evaluation for a batch of states
        # vehicle_node_ids and action_nodes are given as tensors on desired device
        # states is a list of graphs

        #ks = torch.tensor(ks).long()
        #actions = torch.tensor(actions).long()
        #values = torch.tensor(values)
        batched_graph = dgl.batch(states).to(self.device)
        
        batch_x = batched_graph.ndata['feat'].to(self.device)
        batch_e = batched_graph.edata['feat'].to(self.device)

        policy_outputs, value_outputs = self.model(batched_graph, batch_x, batch_e, vehicle_node_ids, masking=True)
        logits = f.softmax(policy_outputs, dim=1)
        probs = Categorical(logits=logits)

        return probs.log_prob(action_nodes), probs.entropy(), value_outputs
    
    def get_state(self):
        free_times = [vehicle.free_time for vehicle in self.darp.vehicles]
        time = np.min(free_times)
        indices = np.argwhere(free_times == time)
        indices = indices.flatten().tolist()

        k = indices[0]
        
        if self.darp.vehicles[k].free_time == 1440:
            return None, None
            
        self.darp.beta(k)
        state, next_vehicle_node = self.darp.state_graph(k, time)

        return state, k, next_vehicle_node

    
    def collect_data(self):
        # Collect data for collect_episodes episodes.
        # Data = s, a, r, s_prime, pi_a, done
        rl_instances = list(range(self.args.num_rl_instances))
        random.shuffle(rl_instances)
        rl_instances_iter = iter(rl_instances)

        for _ in range(self.collect_episodes):
            # Run one episode
            num_instance = next(rl_instances_iter)
            objective = self.darp.reset(num_instance)

            #state_prime, k_prime, next_vehicle_node_prime = self.get_state()

            while self.darp.finish():
                #state, k, next_vehicle_node = state_prime, k_prime, next_vehicle_node_prime
                state, k, next_vehicle_node = self.get_state()

                #action_node, probs = self.darp.predict(state, next_vehicle_node, user_mask=None, src_mask=None)
                action_node, log_prob, current_value = self.select_action(state, next_vehicle_node)
                action = self.darp.node2action(action_node)
                #self.darp.log_probs.append(torch.log(probs.squeeze(0)[action]))
                travel_time, constraint_penalty = self.darp.evaluate_step(k, action)
                transition_cost = travel_time + self.constraint_penalty_alpha * constraint_penalty
                #state_prime, k_prime, next_vehicle_node_prime = self.get_state()
                # Store data
                self.states.append(state)
                self.action_nodes.append(action_node)
                self.costs.append(transition_cost)
                self.log_probs.append(log_prob)
                self.dones.append(not self.darp.finish())
                self.values.append(current_value)
                self.vehicle_node_ids.append(next_vehicle_node)
                #self.data.append((state, action_node, transition_cost, state_prime, prob, not self.darp.finish()))

    
    def compute_returns(self):
        # Compute returns of trajectories
        self.returns = np.zeros(len(self.costs))
        with torch.no_grad():
            for t in reversed(range(len(self.costs))):
                if t == len(self.costs) - 1:
                    next_return = 0
                else:
                    next_return = self.returns[-1]
                nextnonterminal = 0 if self.dones[t] else 1
                self.returns[t] = self.costs[t] + nextnonterminal * next_return
            self.advantages = np.array(self.values) - self.returns # the smaller the returns (cost) the better
    
    

    def data_to_tensors(self):
        # Create torch tensors from collected data.
        states, action_nodes, costs, log_probs, dones, values, vehicle_node_ids, returns, advantages = shuffle_list(self.states,
                                                                                                  self.action_nodes,
                                                                                                  self.costs,
                                                                                                  self.log_probs,
                                                                                                  self.dones,
                                                                                                  self.values,
                                                                                                  self.vehicle_node_ids,
                                                                                                  self.returns,
                                                                                                  self.advantages)
        with torch.no_grad():
            action_nodes_tensor = torch.tensor(action_nodes, device=self.device).long()
            costs_tensor = torch.tensor(costs, device=self.device)
            log_probs_tensor = torch.tensor(log_probs, device=self.device)
            dones_tensor = torch.tensor(dones, device=self.device)
            values_tensor = torch.tensor(values, device=self.device)
            vehicle_node_tensor = torch.tensor(vehicle_node_ids, device=self.device)
            returns_tensor = torch.tensor(returns, device=self.device)
            advantages_tensor = torch.tensor(advantages, device=self.device)
        
        return states, action_nodes_tensor, costs_tensor, log_probs_tensor, dones_tensor, values_tensor, vehicle_node_tensor, returns_tensor, advantages_tensor
    
    def clear_data(self):
        self.states = []
        self.action_nodes = []
        self.costs = []
        self.log_probs = []
        self.dones = []
        self.values = []
        self.vehicle_node_ids = []
        self.returns = None
        self.advantages = None

    def update_model(self):
        # Update model parameters with collected data
        for epoch in range(self.update_epochs):
            states, action_nodes, costs, log_probs, dones, values, vehicle_node_ids, returns, advantages = self.data_to_tensors()
            for start in range(0, len(costs), self.batch_size):
                end = start + self.batch_size
                new_log_probs, entropies, new_values = self.evaluate_policy(states[start:end], vehicle_node_ids[start:end], action_nodes[start:end])
                log_ratio = new_log_probs - log_probs[start:end]
                ratio = log_ratio.exp() # a/b == exp(log(a)-log(b))
                # ... TO CONTINUE

        return None
    
    def train(self):
        # Train policy and value networks on collected data
        
        return None
    
    def save(self, episode, model_name):
        torch.save(self.model.state_dict(), "./model/ppo-{}-{}.pth".format(model_name, episode))
    


