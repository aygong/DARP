import random
import torch
import torch.nn.functional as f

class PPO:
    def __init__(
            self,
            device,
            darp,
            actor_critic_model,
            n_actions,
            gamma,
            lr,
            clip_rate,
            batch_size,
            n_epochs,
            collect_steps,
            num_instances
            ):
        
        self.darp = darp
        self.device = device
        self.model = actor_critic_model
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.clip_rate = clip_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.collect_steps = collect_steps
        self.num_instances = num_instances

    def greedy_action(self, graph, vehicle_node_id):
        # Select action with max probability
        graph = graph.to(self.device)
        k = torch.tensor([vehicle_node_id], device=self.device)
        x = graph.ndata['feat'].to(self.device)
        e = graph.edata['feat'].to(self.device)

        with torch.no_grad():
            policy, _ = self.model(graph, x, e, k, masking=True)
            probs = f.softmax(policy, dim=1)
            a = torch.argmax(probs).item()
            p = probs[a].item()
        return a, p
    
    def select_action(self, graph, vehicle_node_id):
        # Stochastic action selection
        graph = graph.to(self.device)
        k = torch.tensor([vehicle_node_id], device=self.device)
        x = graph.ndata['feat'].to(self.device)
        e = graph.edata['feat'].to(self.device)

        with torch.no_grad():
            policy, _ = self.model(graph, x, e, k, masking=True)
            probs = f.softmax(policy, dim=1)
            a = probs.multinomial(num_samples=1).item()
            p = probs[a].item()
        return a, p
    
    def collect_data(self):
        # Collect data for collect_steps steps
        return None
    
    def compute_returns(self):
        # Compute returns of trajectories
        return None
    
    def make_batch(self):
        # Create batches of data from collected data
        return None
    
    def update_model(self):
        # Update model parameters with collected data
        return None
    
    def train(self):
        # Train policy and value networks on collected data
        return None
    
    def save(self, episode, model_name):
        torch.save(self.model.state_dict(), "./model/ppo-{}-{}.pth".format(model_name, episode))
    



        
    

    #for step in range(train_steps):
        #true_cost = darp.reset(random.choice(num_instances))
        # reset env
        # generate batches of data with current policy. run for ta
        # 


