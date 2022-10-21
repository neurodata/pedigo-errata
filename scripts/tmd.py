#%%
import pickle
from pkg.tmd import TMD
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

neurons_path = "pedigo-errata/data/2022-09-25/neurons.pickle"

#%%
with open(neurons_path, "rb") as f:
    neurons = pickle.load(f)

#%%
neuron = neurons[0]

#%%
def neuron_to_torch_geometric(neuron):
    g = neuron.graph
    treenodes = neuron.nodes
    node_features = treenodes.set_index("node_id")[["x", "y", "z"]]
    adj = nx.to_scipy_sparse_array(g, nodelist=node_features.index)
    adj = adj + adj.T
    row_inds, col_inds = np.nonzero(adj)
    edgelist = np.stack([row_inds, col_inds], axis=1).T
    edge_index = torch.tensor(edgelist, dtype=torch.long)
    x = torch.tensor(node_features.values, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data


def neuron_to_adjacency_and_features(neuron):
    g = neuron.graph
    treenodes = neuron.nodes
    node_features = treenodes.set_index("node_id")[["x", "y", "z"]]
    adj = nx.to_scipy_sparse_array(g, nodelist=node_features.index)
    return adj, node_features


#%%
for i, neuron in enumerate(neurons[:100]):
    print(i, len(neuron.nodes))

#%%
data1 = neuron_to_torch_geometric(neurons[1])
data2 = neuron_to_torch_geometric(neurons[3])

#%%
wass = TMD(data1, data2, w=1.0, L=5)

#%%
from pkg.tmd import TMD
from torch_geometric.datasets import TUDataset

dataset = TUDataset("data", name="MUTAG")
d = TMD(dataset[0], dataset[4], w=1.0, L=4)

#%%
adj1, features1 = neuron_to_adjacency_and_features(neurons[1])
adj2, features2 = neuron_to_adjacency_and_features(neurons[3])
#%%
from sklearn.metrics import pairwise_distances

pdists = pairwise_distances(features1, features2)
pdists.shape

#%%
import seaborn as sns

sns.heatmap(pdists, cmap="RdBu_r", vmin=0)
