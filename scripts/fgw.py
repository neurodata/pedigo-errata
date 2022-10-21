#%%
import pickle
import networkx as nx
import numpy as np
import pandas as pd
from ot import emd
from ot.gromov import fused_gromov_wasserstein, fused_gromov_wasserstein2
from sklearn.metrics import pairwise_distances
from tqdm.notebook import tqdm
import navis

#%%

neurons_path = "pedigo-errata/data/2022-09-25/neurons.pickle"

with open(neurons_path, "rb") as f:
    neurons = pickle.load(f)


skids = neurons.skeleton_id.astype(int)
neurons = {skid: neuron for skid, neuron in zip(skids, neurons)}


#%%
node_data_path = "pedigo-errata/data/2022-09-25/meta_data.csv"

node_data = pd.read_csv(node_data_path, index_col=0)

# node_data_group1 = node_data.query("hemisphere == 'L' and celltype_discrete == 'KCs'")
node_data_group1 = node_data[node_data["name"].str.contains("mPN")]
node_data_group1 = node_data_group1.query("hemisphere == 'L'")
print(len(node_data_group1))
node_data_group2 = node_data[node_data["name"].str.contains("CA-LP")]
node_data_group2 = node_data_group2.query("hemisphere == 'L'")
print(len(node_data_group2))

n_per_group = 3
node_data_group1 = node_data_group1.sample(n_per_group, random_state=2)
node_data_group2 = node_data_group2.sample(n_per_group, random_state=4)
#%%
neurons_group1 = [neurons[skid] for skid in node_data_group1.index]
navis.plot3d(neurons_group1)
#%%
neurons_group2 = [neurons[skid] for skid in node_data_group2.index]
navis.plot3d(neurons_group2)
#%%
group_neurons = neurons_group1 + neurons_group2
navis.plot3d(
    group_neurons, clusters=[0] * len(neurons_group1) + [1] * len(neurons_group2)
)
# %%

from scipy.sparse.csgraph import shortest_path


def neuron_to_adjacency_and_features(neuron):
    g = neuron.graph
    treenodes = neuron.nodes
    node_features = treenodes.set_index("node_id")[["x", "y", "z"]]
    adj = nx.to_scipy_sparse_array(g, nodelist=node_features.index)
    cost = shortest_path(adj, directed=False)
    return cost, node_features


neuron = neurons_group1[1]
cost, features = neuron_to_adjacency_and_features(neuron)

#%%
ax, pos = navis.plot_flat(neuron, color="k")

soma_id = neuron.soma[0]
iloc = features.index.get_loc(soma_id)
navis.plot3d(neuron, color_by=cost[iloc, :], palette="Reds")

#%%
neuron = neurons_group1[0]
neuron.nodes

#%%

neuron1 = neurons_group1[0]
neuron2 = neurons_group1[1]
navis.plot3d([neuron1, neuron2])

#%%


cost1, features1 = neuron_to_adjacency_and_features(neuron1)
cost2, features2 = neuron_to_adjacency_and_features(neuron2)
p = np.full(cost1.shape[0], 1 / cost1.shape[0])
q = np.full(cost2.shape[0], 1 / cost2.shape[0])
pdists = pairwise_distances(features1, features2)
# transport_plan = fused_gromov_wasserstein(
#     pdists,
#     cost1,
#     cost2,
#     p,
#     q,
#     loss_fun="square_loss",
#     alpha=0.5,
#     log=False,
#     verbose=False,
#     numItermax=1000000,
# )
# transport_plan = emd([], [], pdists)
#%%
from ot.bregman import empirical_sinkhorn

transport_plan = empirical_sinkhorn(
    features1.values, features1.values + np.random.normal(features1.values.shape, 1), 1
)

transport_plan.sum(axis=1)
#%%
converged = np.allclose(transport_plan.sum(axis=1), 1 / len(features1))
print("Converged:", converged)

#%%
from scipy.optimize import linear_sum_assignment

row_inds, col_inds = linear_sum_assignment(transport_plan, maximize=True)

#%%
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

fig, axs = plt.subplots(2, 1, figsize=(10, 5))

ax, pos1 = navis.plot_flat(neuron1, color="b", ax=axs[0])
ax, pos2 = navis.plot_flat(neuron2, color="r", ax=axs[1])

n_lines = 200
p_show = n_lines / len(row_inds)
for i, j in zip(row_inds, col_inds):
    if np.random.rand() < p_show:
        node1 = features1.index[i]
        node2 = features2.index[j]
        xy1 = pos1[node1]
        xy2 = pos2[node2]

        con = ConnectionPatch(
            xyA=xy1,
            xyB=xy2,
            coordsA="data",
            coordsB="data",
            axesA=axs[0],
            axesB=axs[1],
            color="grey",
            lw=0.5,
        )
        axs[1].add_artist(con)


#%%
out = navis.plot3d([neuron1, neuron2], inline=False)
import plotly


for i, j in zip(row_inds, col_inds):
    if np.random.rand() < p_show:
        # node1 = features1.index[i]
        # node2 = features2.index[j]

        xyz1 = features1.iloc[i]
        xyz2 = features2.iloc[j]
        plot_data = {
            "x": [xyz1[0], xyz2[0]],
            "y": [xyz1[1], xyz2[1]],
            "z": [xyz1[2], xyz2[2]],
        }
        scatter = plotly.graph_objects.Scatter3d(
            **plot_data,
            mode="lines",
            line=dict(color="grey", width=0.5),
            showlegend=False
        )
        out.add_trace(scatter)
        # out.add_scatter3d(xyz2, color="r", size=10)
out


#%%

total_comparisons = len(group_neurons) * (len(group_neurons) - 1) / 2

pbar = tqdm(total=total_comparisons)

costs = np.zeros((len(group_neurons), len(group_neurons)))
for i, neuron1 in enumerate(group_neurons):
    cost1, features1 = neuron_to_adjacency_and_features(neuron1)
    p = np.full(cost1.shape[0], 1 / cost1.shape[0])
    for j, neuron2 in enumerate(group_neurons[i + 1 :]):
        cost2, features2 = neuron_to_adjacency_and_features(neuron2)
        q = np.full(cost2.shape[0], 1 / cost2.shape[0])

        pdists = pairwise_distances(features1, features2)

        cost = fused_gromov_wasserstein2(
            pdists,
            cost1,
            cost2,
            p,
            q,
            loss_fun="square_loss",
            alpha=0.5,
            log=False,
            verbose=False,
            numItermax=1000000,
        )
        costs[i, j + i + 1] = cost
        pbar.update(1)


costs = costs + costs.T
#%%
import seaborn as sns

sns.heatmap(
    costs, square=True, xticklabels=False, yticklabels=False, cmap="RdBu_r", center=0
)
