# src/rsd/graphs.py

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data, InMemoryDataset
from .utils import PATHS, COORDS, min_img


# ------------------
# Functions
# ------------------

def find_neighbors(pos, is_central, boxsize, cyl_dim):
    """
    Find central galaxies and their neighbors in a cylinder.
    Returns
    central_idx[k]: catalog index of graph k center node
    node_idx[k]: catalog indices of all nodes in graph k
    """
    r_perp_max, s_par_max = cyl_dim
    query_radius=np.hypot(r_perp_max, s_par_max)
    central_idx = np.flatnonzero(is_central)
    # Find neighbors using periodic boundary conditions
    positions = pos % boxsize
    tree = cKDTree(positions, boxsize=boxsize)
    candidates = tree.query_ball_point(
        positions[central_idx], 
        query_radius, 
        workers=n_threads,
    )
    # TODO: need to get nthreads from config here
    
    # Need to mask out the cylinder only
    node_idx = []
    for center, candidate_idx in zip(central_idx, candidates):
        delta = min_img(
            positions[candidate_idx]-positions[center], 
            boxsize,
        )
        
        r_perp = np.linalg.norm(delta[:, :2], axis=1)
        s_par = delta[:, 2]
        inside_cylinder = (
            (r_perp <= r_perp_max)
            & (np.abs(s_par) <= s_par_max)
        )
        nodes = candidate_idx[inside_cylinder]
        # ensures center is at 0
        # TODO: check this in case it already happens automatically
        nodes = nodes[nodes!=center]
        nodes = np.concatenate(([center],nodes))
        node_idx.append(nodes)
    return central_idx, node_idx

def build_graph(
    # center,
    node_idx,
    pos,
    pos_real,
    is_central,
    halo_id,
    boxsize,
    scale
):
    # ---- nodes ----
    # pos wrt center
    delta_rsd = min_img(
        pos - pos[0],
        boxsize,
    )
    delta_real = min_img(
        pos_real - pos_real[0],
        boxsize,
    )
    
    center_mask = np.zeros(n_nodes, dtype=bool)
    center_mask[0] = True
    
    perp_xy = delta_rsd[:, :2]
    r_perp = np.linalg.norm(perp_xy, axis=1)

    s_par = delta_rsd[:, 2]
    z_par = delta_real[:, 2]
    
    delta_true = (z_par - s_par).astype(np.float32)
    node_features = np.column_stack([
            # perp_xy / scale,
            r_perp / scale,
            s_par / scale,
            # center_node,
    ])
    
    # ---- edges ----
    # Create edges between all pairs (no self edges)
    n_nodes = len(node_idx)
    adjacency = ~np.eye(n_nodes, dtype=bool)
    src, dst = np.nonzero(adjacency)

    edge_idx = np.vstack([
        src,
        dst,
    ]).astype(np.int64)

    num = np.sum(perp_xy[src] * perp_xy[dst], axis=1)
    denom = r_perp[src] * r_perp[dst]
    
    # relative angles
    cos_theta = np.zeros(len(src), dtype=np.float32)
    valid_angle = denom > 1e-12 # undefined for small enough transverse sep
    cos_theta[valid_angle] = (
        num[valid_angle] / denom[valid_angle]
    )
    edge_to_center = (center_node[src] | center_node[dst])
    
    r_perp_sq = (
        r_perp[src] ** 2
        + r_perp[dst] ** 2
        - 2.0 * r_perp[src] * r_perp[dst] * cos_theta
    )
    r_perp_sq = np.maximum(r_perp_sq, 0.0)

    delta_perp = perp_xy[src] - perp_xy[dst]
    delta_perp_sq = np.sum(delta_perp**2, axis=1)
    
    delta_z = z_par[src] - z_par[dst] # truth
    delta_dist = np.sqrt(delta_perp_sq + delta_z**2) # truth

    same_halo = halo_id[src] == halo_id[dst]
    
    return Data(
        # nodes
        x=torch.from_numpy(node_features),
        # fixed
        s_par = torch.from_numpy(s_par / scale),
        r_perp = torch.from_numpy(r_perp / scale),
        # training targets
        z_par=torch.from_numpy(z_par / scale),
        delta_true=torch.from_numpy(delta_true / scale),
        is_central=torch.from_numpy(is_central),
        halo_id=torch.from_numpy(halo_id),
        # edges
        edge_index=torch.from_numpy(edge_idx),
        edge_to_center=torch.from_numpy(edge_to_center),
        edge_angle=torch.from_numpy(cos_theta[:, None]),
        edge_rperp_sq=torch.from_numpy(
            delta_perp_sq / scale**2
        ),
        edge_dist = torch.from_numpy(delta_dist / scale),
        same_halo = torch.from_numpy(same_halo),
        # to map back to catalog
        center_id=torch.tensor([node_idx[0]], dtype=torch.long),
        n_id=torch.from_numpy(node_idx),
    )

# TODO: instead of taking cat and boxsize, pass thru the mockid
def generate_graphs(cat, boxsize, cyl_dim):
    """Build all non-singlet graphs for one mock catalog."""
    # TODO: i want to input redshift!!
    pos = cat["GalaxyPos"] % boxsize
    pos_real = cat["RealGalaxyPos"] % boxsize
    is_central = cat["is_central"].astype(bool)
    halo_id = cat["halo_id"]
    central_idx, neighbor_idx = find_neighbors(
        pos=pos,
        is_central=is_central,
        boxsize=boxsize,
        cyl_dim=cyl_dim,
    )
    scale = float(cyl_dim[0])
    graphs = []

    for nodes in neighbor_idx:
        if len(nodes) < 2:
            continue
        graphs.append(
            build_graph(
                node_idx=nodes,
                pos=pos[nodes],
                pos_real=pos_real[nodes],
                is_central=is_central[nodes].astype(bool),
                halo_id=halo_id[nodes],
                boxsize=boxsize,
                scale=scale,
            )
        )
    return graphs
    
def save_dataset(graphs, path):
    """save one collated PyG dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    InMemoryDataset.save(graphs, path)
    return path, len(graphs)

# ------------------
# Class
# ------------------

class GraphDataset(InMemoryDataset):
    def __init__(self, path, transform=None):
        super().__init__(
            root=None,
            transform=transform,
        )
        self.load(path)
