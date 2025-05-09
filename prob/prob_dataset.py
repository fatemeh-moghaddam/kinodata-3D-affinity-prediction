import kinodata
from kinodata.data import KinodataDocked
from kinodata.types import *
from kinodata.model import ComplexTransformer

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch

import json
from pathlib import Path
import os
from typing import Any
from functools import partial
from collections import defaultdict


# import matplotlib.pyplot as plt
# import seaborn as sns

import pandas as pd
import numpy as np
from tqdm import tqdm
import random



_DATA = Path.cwd().parent / "data"
CPU_COUNT = 12


def make_probing_x(
    data: KinodataDocked,
    gnn_model: ComplexTransformer,
    num_workers: int = CPU_COUNT,
    batch_size: int = 1,
    shuffle: bool = False,
    graph_pool: bool = True
    ) -> Tensor:
    """
    Create a probing dataset by running the model on the original dataset.
    This will create multiple datasets, based on the number and of attnetion blocks.
    The probing dataset will be saved in the probing directory.
    The target of prob, is not specified to here.
    The probing dataset will be saved in the probing directory.
    """
    probing_dir = _DATA / "probing"
    probing_dir.mkdir(exist_ok=True, parents=True)

    dataset = DataLoader(data, batch_size= batch_size, shuffle=shuffle, num_workers=num_workers) 
    # the mapping of batch index to graph ident
    graph_idents = dataset.ident.tolist()

    # Forward pass to get the X
    gnn_model.eval()
    with torch.no_grad():
        out_graphs, intermediate_node_reprs, intermediate_edge_reprs, prior_readout = gnn_model(dataset)
    
    layers = intermediate_node_reprs.keys()
    for layer, layer_name in enumerate(layers):
        layer_dir = probing_dir / layer_name
        layer_dir.mkdir(exist_ok=True, parents=True)

        node_repr, batch_index = intermediate_node_reprs[layer]
        # edge_repr, edge_index = intermediate_edge_reprs[layer]
        """
            node_repr: Tensor  # shape: [total_nodes_all_graphs, 256]
            batch_index: LongTensor  # shape: [total_nodes_all_graphs], values ∈ [0, 41238]
        """

        # Graph-level
        if graph_pool:
            graph_repr = {}
            # intermediate_graph_reprs = {}
            # Get the graph level representation
            # intermediate_graph_reprs[layer] = gnn_model.aggr(node_repr, batch_index)
            graph_repr[ident] = gnn_model.aggr(node_repr, batch_index)
            """"
                graph_repr: Tensor  # shape: [41238, 256]
            """
            # Save the graph level representation
            # torch.save(intermediate_graph_reprs[layer], layer_dir / f"graph_repr.pt")
            torch.save(graph_repr, layer_dir / f"graph_level_repr.pt")

        # Node-level
        else:
            graph_list = unbatch(node_repr, batch_index)
            node_reprs = {}
            # Save the node level representation
            for i, graph in enumerate(graph_list):
                # attach graph_ident
                node_reprs[ident] = graph
            torch.save(node_reprs, layer_dir / f"node_level_reprs.pt")
    
    # Separate by layer


    return dataset



def make_probing_y(target, mapper: pd.DataFrame) -> np.ndarray:
    """
    Get the probing target for the dataset.
    The target is the output of the model.
    """

    
    return y

