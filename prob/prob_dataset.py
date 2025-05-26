import kinodata
from kinodata.data import KinodataDocked
from kinodata.types import *
from kinodata.model import ComplexTransformer

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch, unbatch_edge_index

import json
from pathlib import Path
import os
from typing import Any, List, Dict, Tuple, Literal

import pandas as pd
import numpy as np
from tqdm import tqdm



_DATA = Path.cwd().parent / "data"
CPU_COUNT = 12

# store a list of GraphReprs objects per layer


class GraphReprs:
    def __init__(self, ident: int):
        self.ident = ident
        self.node_repr = {}     # layer_name -> Tensor [num_nodes, hidden_dim]
        self.graph_repr = {}    # layer_name -> Tensor [hidden_dim]
        self.edge_repr = {}     # layer_name -> Tensor [num_edges, hidden_dim]
        self.edge_index = {}    # layer_name -> Tensor [2, num_nodes]
        self.properties = {}    # dict of properties
        



    def add_property(self, property_name: str, property_value: Any):
        if property_name in self.properties:
            raise ValueError(f"Property {property_name} already exists.")
        self.properties[property_name] = property_value


    @property
    def mw(self):
        """
        returns the molecular weight of the graph, using ident and a pre-computed mapping.
        """
        return self.properties["mw"]

    @property
    def num_Hbonds(self):
        """
        Computes and returns the number of Hydrogen bonds in the graph.
        """
        return self.properties["num_Hbonds"]


    def get_property(self, name: str):
        return self.properties[name]


    def __repr__(self):
        return f"<KinodataDocked ident= {self.ident}, layers={list(self.node_repr.keys())}>"



class ProbingDataset:
    def __init__(self, 
                 orig_dataset: KinodataDocked, 
                 gnn_model: ComplexTransformer,
                 gnn_model_ckpt: str,
                 save_dir: Path = _DATA / "probing",
                 num_workers: int = CPU_COUNT,
                 graph_level: bool = True,
                 batch_size: int = 1,
                 shuffle: bool = False
                 ):
        self.orig_dataset = orig_dataset
        self.gnn_model = gnn_model
        self.probing_dir = save_dir
        self.probing_dir.mkdir(exist_ok=True, parents=True)
        self.num_workers = num_workers
        self.graph_level = graph_level
        if graph_level:
            self.graph_pool = gnn_model.aggr
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gnn_model_ckpt = gnn_model_ckpt
        self.loaded_gnn_model = None

    
    @property
    def loaded_gnn_model(self):
        """
        Load the model from the checkpoint
        """
        if self.loaded_gnn_model is not None:
            return self.loaded_gnn_model
        ckp = torch.load(self.gnn_model_ckpt, map_location="cpu")
        model = self.gnn_model
        model.load_state_dict(ckp["state_dict"])
        return model.eval()
        

    def __call__(self) -> List[GraphReprs]:
        """
        Run the ProbingDataset to get a specific probing dataset.
        This will load the model from a checkpoint, compute the representations, and return a list of GraphReprs objects.
        """
        # Do I separate the graph-level computation from the node-level computation, here?
        return self.compute_reprs()


    # ───────────────────────────────────────────────────────
    # Unbatching functions, NOT TESTED YET
    # ───────────────────────────────────────────────────────

    def _unbatch(self,
                        # graph: GraphReprs,
                        node_repr: Tensor,
                        batch_index: torch.LongTensor,
                        edge_index: Tensor = None,
                        edge_repr: Tensor = None,
                       )-> Dict:
        """
            Process the node representations for a batch of KinodataDocked graphs.
            That means attaching the graph ident to the node representation with , 
            and pooling the node representation to a graph representation if graph_level is True.
            args:
                ident_list: List of KinodataDocked.ident
                node_repr: Dictionary of node representations
                    - layer name as key: layer_1, layer_2, layer_3, prior_readout
                    - tensor of shape [total_nodes_all_graphs, 256] as value
                batch_index: tensor of shape [total_nodes_all_graphs]
            returns:
                list of Tensor
        """
        if self.graph_level:
            return list(self.graph_pool(node_repr, batch_index))    

        # unbatch the node representation
        node_reprs = unbatch(node_repr, batch_index)
        edge_indices = unbatch_edge_index(edge_index, batch_index)    
        edge_reprs = self._slice_edge_reprs(edge_repr, edge_indices, batch_index)
        
        return node_reprs, edge_reprs, edge_indices


    # for atom-level probing
    def _slice_edge_reprs(self,
                        edge_repr: Tensor,
                        edge_indices: List[Tensor],
                       )-> Dict:
        ''' edge_reprs/edge_features has to be done manually, because the edge_feature is not in the same order as the node features
            so batch cannot be used to unbatch the edge features, cause that's node based '''
        # assumes all edges whose source nodes share the same batch_id belong to the same graph
    	# edge_indices is a list of edge_index(2, num_edges) tensors, where each tensor belongs to one graph
        ## figure out where each graph’s edges start and end by keeping track of the cumulative sum of number of edges
        edge_pointer = [0]
        for graph_ei in edge_indices:
            ei_start_point = edge_pointer[-1]  
            ei_count_point = graph_ei.size(1)     # ei.size = ([2, number_of_edges_in_batch])
            edge_pointer.append(ei_start_point + ei_count_point)
        # slice the edge_reprs tensor of graphs in the batch to get the unbatched edge features
        edge_reprs = [edge_repr[edge_pointer[i]:edge_pointer[i+1]] for i in range(len(edge_pointer)-1)]
        return edge_reprs



    # ───────────────────────────────────────────────────────
    # Main function/Pipeline
    # ───────────────────────────────────────────────────────

    def compute_reprs(self) -> List[GraphReprs]:
        """
        Compute the representations for a batch of KinodataDocked graphs, with a forward pass of the model.
        returns a list of GraphReprs objects for each graph in the batch.
        """
        graph_list = []
        data_module = DataLoader(self.orig_dataset, batch_size= self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        

        with torch.no_grad():
            for data in tqdm(data_module):
                batch_graph_idents = data.ident.tolist()
                out_graphs, intermediate_node_reprs, intermediate_edge_reprs, prior_readout = self.loaded_gnn_model(data)
                """
                out_graphs: Tensor                          # shape: [batch_size, 1]
                intermediate_node_reprs: Dict[str, Tensor]  # layer_name -> Tensor [total_nodes_all_graphs, 256]
                intermediate_edge_reprs: Dict[str, Tensor]  # layer_name -> Tensor [total_edges_all_graphs, 256]
                prior_readout: Tensor                       # shape: [batch_size, 256]
                """
                graph_reprs_list = list(prior_readout)
                
                # process the representations per layer
                for layer_name in intermediate_node_reprs.keys():
                    node_repr, batch_index = intermediate_node_reprs[layer_name]
                    edge_repr, edge_index = intermediate_edge_reprs[layer_name]
                    # process the representations
                    if self.batch_size == 1:
                        # Batch size is 1, so we can directly use the node_repr and edge_repr
                        processed_reprs = (node_repr, edge_repr, edge_index)
                    elif self.graph_level:
                        # Graph level, so we just need to pool the node representations 
                        # basically, just using the graph_pool here
                        # and the edge_index is not needed
                        processed_reprs = self._unbatch(node_repr, batch_index)
                    else:
                        # Unbatch
                        processed_reprs = self._unbatch(node_repr, batch_index, edge_index, edge_repr)
                    # create a list of GraphReprs for each graph in the batch
                    for i, ident in enumerate(batch_graph_idents):
                        # ident = batch_graph_idents[i]
                        graph = GraphReprs(ident)
                        graph.graph_repr['prior_readout'] = graph_reprs_list[i]

                        if self.graph_level:
                            # add the graph representation
                            graph.graph_repr[layer_name] = processed_reprs[i]
                        else:   
                            # unbatched
                            node_reprs, edge_reprs, edge_indices = processed_reprs
                            graph.node_repr[layer_name] = node_reprs[i]
                            graph.edge_repr[layer_name] = edge_reprs[i]
                            graph.edge_index[layer_name] = edge_indices[i]

                        # add the properties?
                        # add the graph to the list
                        graph_list.append(graph)

        return graph_list



    # ───────────────────────────────────────────────────────
    # From GraphReprs list to Numpy
    # ───────────────────────────────────────────────────────

    def get_X(self,
            graphs: List['GraphReprs'],
            layer_name: str,
            level: Literal["graph", "node"] = "graph"
            ) -> np.ndarray:
        """
        Extract feature matrix X from GraphReprs objects for a given layer and level.

        Returns:
            X: np.ndarray, shape [n_samples, hidden_dim]
        """
        X_list = []

        for g in graphs:
            if level == "graph":
                x = g.graph_repr[layer_name].detach().cpu().numpy()
                X_list.append(x)
            elif level == "node":
                x = g.node_repr[layer_name].detach().cpu().numpy()
                X_list.append(x)
            else:
                raise ValueError("level must be 'graph' or 'node'")

        return np.vstack(X_list)



    def get_y(self,
            graphs: List['GraphReprs'],
            target: str,
            level: Literal["graph", "node"] = "graph",
            ) -> np.ndarray:
        """
        Extract target vector y from GraphReprs objects for a given level.

        Returns:
            y: np.ndarray, shape [n_samples]
        """
        y_list = []

        for g in graphs:
            y = g.get_property(target)
            if level == "graph":
                y_list.append(y)
            elif level == "node":
                # y should be a 2D array with shape (NUM_GRAPHS, NUM_NODES)
                pass 
            else:
                raise ValueError("level must be 'graph' or 'node'")

        return np.array(y_list)


