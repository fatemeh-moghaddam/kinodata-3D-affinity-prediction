import kinodata
from kinodata.data import KinodataDocked
from kinodata.types import *
from kinodata.model import ComplexTransformer, RegressionModel
from kinodata.configuration import Config

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
# class GraphReprs:
#     def __init__(self, ident: int):
#         self.ident = ident
#         self.node_repr = {}     # layer_name -> Tensor [num_nodes, hidden_dim]
#         self.graph_repr = {}    # layer_name -> Tensor [hidden_dim]
#         self.edge_repr = {}     # layer_name -> Tensor [num_edges, hidden_dim]
#         self.edge_index = {}    # layer_name -> Tensor [2, num_nodes]
#         self.properties = {}    # dict of properties
        



#     def add_property(self, property_name: str, property_value: Any):
#         if property_name in self.properties:
#             raise ValueError(f"Property {property_name} already exists.")
#         self.properties[property_name] = property_value


#     @property
#     def mw(self):
#         """
#         returns the molecular weight of the graph, using ident and a pre-computed mapping.
#         """
#         return self.properties["mw"]

#     @property
#     def num_Hbonds(self):
#         """
#         Computes and returns the number of Hydrogen bonds in the graph.
#         """
#         return self.properties["num_Hbonds"]


#     def get_property(self, name: str):
#         return self.properties[name]


#     def __repr__(self):
#         return f"<KinodataDocked ident= {self.ident}, layers={list(self.node_repr.keys())}>"

def save_fold_tensor(tensor: torch.Tensor, output_dir: Path, filename: str):
    """
    Save a tensor to a file in the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / filename
    torch.save(tensor, out_file)
    return out_file


# A function instead of a class?
def run_fold(ds: KinodataDocked,
             gnn_model: RegressionModel,
             config: Config):
    """
    Run a fold on a given KinodataDocked split and write per-layer [graph-level for now]
    representations as a .pt file.

    Expects in `config`:
      - batch_size: int
      - output_fold_dir: Path
      - split_index: int | str
      - graph_level: bool (must be True)
      - dtype (optional): torch.dtype or string ('float32'/'float16')
      - expected_layer_dims (optional): dict[layer_name] = d_k for validation
      - validate (optional): bool to enable asserts
    """

    fold_size = len(ds)

    # Per-layer buffers as lists for concatenation; plus prior_readout
    layer_bufs: Dict[str, List[torch.Tensor]] = {}
    prior_buf: List[torch.Tensor] = []
    idents: List[int] = []


    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing graph representations"):
            # IDs
            batch_idents = batch.ident.tolist()
            idents.extend(batch_idents)


            if not config.graph_level:
                raise NotImplementedError("Only graph-level probing is implemented for now.")
                _ , intermediate_node_reprs, intermediate_edge_reprs, _ = gnn_model(batch)

            # Forward
            _ , intermediate_node_reprs, _ , prior_readout = gnn_model(batch)

            # Pool each layer with the model's aggregator
            # intermediate_node_reprs: {layer_name: (node_repr, batch_index)}
            for layer_name, (node_repr, batch_index) in intermediate_node_reprs.items():
                graph_repr = gnn_model.aggr(node_repr, batch_index).detach().cpu()

                # Create a buffer for each layer
                if layer_name not in layer_bufs:
                    layer_bufs[layer_name] = []

                # Possible memory optimization
                if dtype_output := config.dtype:
                    graph_repr = graph_repr.to(dtype_output)

                layer_bufs[layer_name].append(graph_repr)
                prior_buf.append(prior_readout.detach().cpu())
        
        # Concatenate per-layer and prior_readout
        layers_cat: Dict[str, torch.Tensor] = {}
        for layer_name, chunks in layer_bufs.items():
            layers_cat[layer_name] = torch.cat(chunks, dim=0)  # [N_fold, d_k]

        prior_cat = torch.cat(prior_buf, dim=0)  # [N_fold, d_pr]
        ids_tensor = torch.tensor(idents, dtype=torch.long)

        # Checks
        assert ids_tensor.shape[0] == fold_size, "IDs tensor size mismatch with fold size"
        assert prior_cat.shape[0] == fold_size, "Prior tensor size mismatch with fold size"
        for layer_name, layer_cat in layers_cat.items():
            assert layer_cat.shape[0] == fold_size, f"Layer {layer_name} tensor size mismatch with fold size"

        # Save to disk separately
        for layer_name, layer_cat in layers_cat.items():
            save_fold_tensor(layer_cat, config.output_fold_dir, f"layer_{layer_name}_{config.split_index}.pt")
        save_fold_tensor(prior_cat, config.output_fold_dir, f"prior_{config.split_index}.pt")
        save_fold_tensor(ids_tensor, config.output_fold_dir, f"ids_{config.split_index}.pt")

    return

#######################################
class ProbingDataset:
    def __init__(self, 
                 dataset: KinodataDocked, 
                 gnn_model: ComplexTransformer,
                 gnn_model_ckpt: str = None,
                 save_dir: Path = _DATA / "probing",
                 num_workers: int = CPU_COUNT,
                 graph_level: bool = True,
                 batch_size: int = 1,
                 shuffle: bool = False
                 ):
        self.dataset = dataset
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
        self._loaded_gnn_model = None

    
    @property
    def loaded_gnn_model(self):
        """
        Load the model from the checkpoint
        """
        if self._loaded_gnn_model is not None:
            return self._loaded_gnn_model
        ckp = torch.load(self.gnn_model_ckpt, map_location="cpu")
        model = self.gnn_model
        model.load_state_dict(ckp["state_dict"])
        self._loaded_gnn_model = self.gnn_model.eval()
        return self._loaded_gnn_model
        


    def __call__(self) -> List[GraphReprs]:
        """
        Run the ProbingDataset to get a specific probing dataset.
        This will load the model from a checkpoint, compute the representations, and return a list of GraphReprs objects.
        """
        # Do I separate the graph-level computation from the node-level computation, here?
        return self.compute_graph_reprs() if self.graph_level else self.compute_atom_reprs()




    # ───────────────────────────────────────────────────────
    # Main function/Pipeline
    # ───────────────────────────────────────────────────────


    def compute_graph_reprs(self) -> List[GraphReprs]:
        """
        Computes per-layer graph-level representations for all graphs in the dataset.
        """
        ## Do I create the test dataset here?
        loader = DataLoader(self.dataset, batch_size=1, shuffle=self.shuffle, num_workers=self.num_workers)
        model = self.loaded_gnn_model

        output: List[GraphReprs] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing graph representations"):
                # batch is a single KinodataDocked graph, as a HeteroDataBatch
                ident = batch.ident.item()
                _ , intermediate_node_reprs, intermediate_edge_reprs, prior_readout = model(batch)
                #             """
                #             out_graphs: Tensor                          # shape: [batch_size, 1]
                #             intermediate_node_reprs: Dict[str, Tensor]  # layer_name -> Tensor [total_nodes_all_graphs, 256]
                #             intermediate_edge_reprs: Dict[str, Tensor]  # layer_name -> Tensor [total_edges_all_graphs, 256]
                #             prior_readout: Tensor                       # shape: [batch_size, 256]
                #             """

                graph = GraphReprs(ident)

                for layer_name in intermediate_node_reprs.keys():
                    node_repr, batch_index = intermediate_node_reprs[layer_name]
                    edge_repr, edge_index = intermediate_edge_reprs[layer_name]

                    if self.graph_level:
                        # add the graph representation
                        graph.graph_repr[layer_name] = self.graph_pool(node_repr, batch_index)
                    else:
                        graph.node_repr[layer_name] = node_repr
                        graph.edge_repr[layer_name] = edge_repr
                        graph.edge_index[layer_name] = edge_index
                # add the prior readout
                graph.graph_repr['prior_readout'] = prior_readout
                    
                output.append(graph)

        return output



########################### DEPRECATED ###########################

    # # ───────────────────────────────────────────────────────
    # # Unbatching functions, NOT TESTED YET
    # # ───────────────────────────────────────────────────────

    # def _unbatch(self,
    #                     # graph: GraphReprs,
    #                     node_repr: Tensor,
    #                     batch_index: torch.LongTensor,
    #                     edge_index: Tensor = None,
    #                     edge_repr: Tensor = None,
    #                    )-> Dict:
    #     """
    #         Process the node representations for a batch of KinodataDocked graphs.
    #         That means attaching the graph ident to the node representation with , 
    #         and pooling the node representation to a graph representation if graph_level is True.
    #         args:
    #             ident_list: List of KinodataDocked.ident
    #             node_repr: Dictionary of node representations
    #                 - layer name as key: layer_1, layer_2, layer_3, prior_readout
    #                 - tensor of shape [total_nodes_all_graphs, 256] as value
    #             batch_index: tensor of shape [total_nodes_all_graphs]
    #         returns:
    #             list of Tensor
    #     """
    #     if self.graph_level:
    #         return list(self.graph_pool(node_repr, batch_index))    

    #     # unbatch the node representation
    #     node_reprs = unbatch(node_repr, batch_index)
    #     edge_indices = unbatch_edge_index(edge_index, batch_index)    
    #     edge_reprs = self._slice_edge_reprs(edge_repr, edge_indices, batch_index)
        
    #     return node_reprs, edge_reprs, edge_indices


    # # for atom-level probing
    # def _slice_edge_reprs(self,
    #                     edge_repr: Tensor,
    #                     edge_indices: List[Tensor],
    #                    )-> Dict:
    #     ''' edge_reprs/edge_features has to be done manually, because the edge_feature is not in the same order as the node features
    #         so batch cannot be used to unbatch the edge features, cause that's node based '''
    #     # assumes all edges whose source nodes share the same batch_id belong to the same graph
    # 	# edge_indices is a list of edge_index(2, num_edges) tensors, where each tensor belongs to one graph
    #     ## figure out where each graph’s edges start and end by keeping track of the cumulative sum of number of edges
    #     edge_pointer = [0]
    #     for graph_ei in edge_indices:
    #         ei_start_point = edge_pointer[-1]  
    #         ei_count_point = graph_ei.size(1)     # ei.size = ([2, number_of_edges_in_batch])
    #         edge_pointer.append(ei_start_point + ei_count_point)
    #     # slice the edge_reprs tensor of graphs in the batch to get the unbatched edge features
    #     edge_reprs = [edge_repr[edge_pointer[i]:edge_pointer[i+1]] for i in range(len(edge_pointer)-1)]
    #     return edge_reprs




    # def compute_reprs(self,
    #                   ) -> List[GraphReprs]:
    #     """
    #     Compute the representations for a batch of KinodataDocked graphs, with a forward pass of the model.
    #     returns a list of GraphReprs objects for each graph in the batch.
    #     """
    #     graph_list = []
    #     data_module = DataLoader(self.dataset, batch_size= self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        

    #     with torch.no_grad():
    #         for data in tqdm(data_module):
    #             batch_graph_idents = data.ident.tolist()
    #             _ , intermediate_node_reprs, intermediate_edge_reprs, prior_readout = self.loaded_gnn_model(data)
    #             """
    #             out_graphs: Tensor                          # shape: [batch_size, 1]
    #             intermediate_node_reprs: Dict[str, Tensor]  # layer_name -> Tensor [total_nodes_all_graphs, 256]
    #             intermediate_edge_reprs: Dict[str, Tensor]  # layer_name -> Tensor [total_edges_all_graphs, 256]
    #             prior_readout: Tensor                       # shape: [batch_size, 256]
    #             """
    #             graph_reprs_list = list(prior_readout)
                
    #             # process the representations per layer
    #             for layer_name in intermediate_node_reprs.keys():
    #                 node_repr, batch_index = intermediate_node_reprs[layer_name]
    #                 edge_repr, edge_index = intermediate_edge_reprs[layer_name]
    #                 # process the representations
    #                 if self.batch_size == 1:
    #                     # Batch size is 1, so we can directly use the node_repr and edge_repr
    #                     processed_reprs = (node_repr, edge_repr, edge_index)
    #                 elif self.graph_level:
    #                     # Graph level, so we just need to pool the node representations 
    #                     # basically, just using the graph_pool here
    #                     # and the edge_index is not needed
    #                     processed_reprs = self._unbatch(node_repr, batch_index)
    #                 else:
    #                     # Unbatch
    #                     processed_reprs = self._unbatch(node_repr, batch_index, edge_index, edge_repr)
    #                 # create a list of GraphReprs for each graph in the batch
    #                 for i, ident in enumerate(batch_graph_idents):
    #                     # ident = batch_graph_idents[i]
    #                     graph = GraphReprs(ident)
    #                     graph.graph_repr['prior_readout'] = graph_reprs_list[i]

    #                     if self.graph_level:
    #                         # add the graph representation
    #                         graph.graph_repr[layer_name] = processed_reprs[i]
    #                     else:   
    #                         # unbatched
    #                         node_reprs, edge_reprs, edge_indices = processed_reprs
    #                         graph.node_repr[layer_name] = node_reprs[i]
    #                         graph.edge_repr[layer_name] = edge_reprs[i]
    #                         graph.edge_index[layer_name] = edge_indices[i]

    #                     # add the properties?
    #                     # add the graph to the list
    #                     graph_list.append(graph)

    #     return graph_list



    # ───────────────────────────────────────────────────────
    # From GraphReprs list to Numpy
    # ───────────────────────────────────────────────────────
    # moved to prob/helpers.py