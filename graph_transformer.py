import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from graph_transformer_edge_layer import GraphTransformerLayer

class GraphTransformerNet(nn.Module):
    def __init__(self, 
                 device,
                 num_users,
                 num_vehicles,
                 target_seq_len,
                 num_node_feat,
                 num_edge_feat,
                 d_model=512,
                 num_layers=6,
                 num_heads=8,
                 d_k=64,
                 d_v=64,
                 d_ff=2048,
                 dropout=0.1,
                 #in_feat_dropout=0.0,
                 layer_norm=False,
                 batch_norm=True,
                 lap_pos_enc=False,
                 residual=True
                 ):
        
        super().__init__()

        #num_atom_type = net_params['num_atom_type']
        #num_bond_type = net_params['num_bond_type']
        #hidden_dim = net_params['hidden_dim']
        #num_heads = net_params['n_heads']
        #out_dim = net_params['out_dim']
        #in_feat_dropout = net_params['in_feat_dropout']
        #dropout = net_params['dropout']
        #n_layers = net_params['L']
        #self.readout = net_params['readout']
        #self.layer_norm = net_params['layer_norm']
        #self.batch_norm = net_params['batch_norm']
        #self.residual = net_params['residual']
        #self.edge_feat = net_params['edge_feat']
        #self.device = net_params['device']
        #self.lap_pos_enc = net_params['lap_pos_enc']
        #self.wl_pos_enc = net_params['wl_pos_enc']
        #max_wl_role_index = 37 # this is maximum graph size in the dataset

        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.device = device
        self.lap_pos_enc = lap_pos_enc
        self.residual = residual
        
        if self.lap_pos_enc:
            #pos_enc_dim = net_params['pos_enc_dim']
            #self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
            raise NotImplementedError()
        
        
        self.embedding_h = nn.Linear(num_node_feat, d_model)

        self.embedding_e = nn.Linear(num_edge_feat, d_model)
        
        #self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(d_model, d_model, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(num_layers) ]) 
        #self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        #self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem 
        self.MLP_layer = nn.Linear(2*d_model, 1)       
        
    def forward(self, g, h, e, vehicle_node_id, h_lap_pos_enc=None):

        # input embedding
        h = self.embedding_h(h)
        #h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            #h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            #h = h + h_lap_pos_enc
            raise NotImplementedError()
        
        e = self.embedding_e(e)   
        
        # transformer layers
        for conv in self.layers:
            h, e = conv(g, h, e)
        #g.ndata['h'] = h
        
        #if self.readout == "sum":
        #    hg = dgl.sum_nodes(g, 'h')
        #elif self.readout == "max":
        #    hg = dgl.max_nodes(g, 'h')
        #elif self.readout == "mean":
        #    hg = dgl.mean_nodes(g, 'h')
        #else:
        #    hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        
        pairs = []
        for neighbor in g.successors(vehicle_node_id):
            pair = torch.cat([h[vehicle_node_id], h[neighbor]], dim=1) # shape: (batch_size, 2*d_model)
            pairs.append(pair)

        pairs = torch.stack(pairs, dim=1)
        # also return value
        return self.MLP_layer(hg)
        
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss