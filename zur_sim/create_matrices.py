"""
Cost-matrix routing prototype (single snapshot)
===============================================
Standalone exploratory script that prototypes the traffic-aware routing later
generalised into the ``TransitionMatrix`` class (see transition_matrix.py). It
loads one traffic snapshot (2026-01-08 12:00), aggregates detector counts per
station, projects the coordinates to lat/lon, and builds a node-to-node cost
matrix that blends squared distance and destination traffic (weights
``alpha``/``beta``). The matrix is sparsified to each node's ``k`` nearest
neighbours, then Dijkstra finds the cheapest path between two hard-coded nodes,
which is finally plotted.

This module runs its logic at import time (top-level statements, not functions)
and is kept mainly as a reference/experiment; the reusable version lives in
transition_matrix.py.
"""

import socket
if socket.gethostname() == 'berttrainer-large':
    from datasource import DataSource
    from plots import plot
else:
    from zur_sim.datasource import DataSource
    from zur_sim.plots import plot

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import networkx as nx

ds = DataSource()
df = ds.df
df = df.loc[df.AnzFahrzeuge.notna()]
df_all =  df.loc[df.MessungDatZeit == '2026-01-08T12:00:00']

df = df_all[['ZSID', 'ZSName', 'EKoord', 'Hoehe',
       'NKoord',  'AnzFahrzeuge', 'MessungDatZeit']].groupby('ZSID').agg(
            {'AnzFahrzeuge' : 'mean',
            'EKoord':'mean',
            'NKoord': 'mean',
            'ZSName': 'first',
             'Hoehe': 'first',
             'MessungDatZeit': 'first'}
    ).reset_index()


gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.EKoord, df.NKoord),
        crs="EPSG:2056"
    )

    # convert to lat/lon (required for Plotly)
gdf = gdf.to_crs(4326)
gdf["lat"] = gdf.geometry.y
gdf["lon"] = gdf.geometry.x
nodes = gdf[['lon', 'lat', 'ZSID', 'AnzFahrzeuge']].set_index('ZSID')
nodes.columns = ['x', 'y', 'AnzFahrzeuge']
n_nodes = len(nodes)

# Fully connected transition matrix (row sums to 1)
transition_matrix = np.ones((n_nodes, n_nodes)) #/ (n_nodes - 1)
np.fill_diagonal(transition_matrix, 0)  # cannot stay at the same node

# Extract coordinates as arrays
coords = nodes[["x","y"]].values
traffic = nodes["AnzFahrzeuge"].values

# Traffic cost = moving to a node with high traffic is costly
traffic_matrix = np.tile(traffic, (n_nodes, 1))
# traffic_matrix[i,j] = traffic at node j
# optionally set diagonal to 0 or np.inf
np.fill_diagonal(traffic_matrix, 0)

# Compute distance matrix
distance_matrix = cdist(coords, coords, metric="euclidean")
distance_matrix  = distance_matrix**2
# Optional: set diagonal to infinity (no self-loop)
np.fill_diagonal(distance_matrix, 0)





distance_norm = distance_matrix / distance_matrix.max()
traffic_norm = traffic_matrix / traffic_matrix.max()


alpha = 0.8   # weight distance
beta  = 0.2   # weight traffic

total_cost_matrix = alpha * distance_norm + beta * traffic_norm

# only allow transition to k nearest neighbors
k = 10  # number of nearest neighbors
# Create a sparse cost matrix initialized with infinities
sparse_cost_matrix = np.full(distance_matrix.shape, np.inf)
for i in range(distance_matrix.shape[0]):
    # indices of k closest nodes
    nearest_idx = np.argsort(distance_matrix[i])[:k]
    # copy total_cost values only for these neighbors
    sparse_cost_matrix[i, nearest_idx] = total_cost_matrix[i, nearest_idx]

# Compute cheapest path

# total_cost_matrix is your n_nodes x n_nodes numpy array
G = nx.from_numpy_array(sparse_cost_matrix, create_using=nx.DiGraph)
# G = nx.from_numpy_array(total_cost_matrix, create_using=nx.DiGraph)
# DiGraph because cost[i,j] may not equal cost[j,i] if traffic is directional
source = 0
target = 30

path = nx.dijkstra_path(G, source=source, target=target, weight="weight")
total_cost = nx.dijkstra_path_length(G, source=source, target=target, weight="weight")

print("Cheapest path:", path)
print("Total cost:", total_cost)



import matplotlib.pyplot as plt

x = nodes["x"].values
y = nodes["y"].values

plt.figure(figsize=(6,6))
plt.scatter(x, y, c='blue')

# plot the cheapest path
px_path = [x[i] for i in path]
py_path = [y[i] for i in path]
plt.plot(px_path, py_path, c='red', lw=2, marker='o')

for i, txt in enumerate(range(len(x))):
    plt.text(x[i]+0.1, y[i]+0.1, str(i))

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Cheapest path from node {} to {}".format(source, target))
plt.show()
