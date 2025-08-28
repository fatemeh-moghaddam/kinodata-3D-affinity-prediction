'''
This module includes run_fold, aggregate_fold, and helpers of these functions.
'''
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


##### These need to be moved to io at some point
def save_out_tensor(tensor: torch.Tensor, output_dir: Path, filename: str, fold: int|None):
    """
    Save a tensor to a file in the output directory.
    """
    if fold is not None:
        output_dir = output_dir / str(fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / filename
    torch.save(tensor, out_file)
    return out_file


def load_tensor(dir: Path, filename: str) -> torch.Tensor:
    """
    Load a tensor from a file in the directory.
    """
    tensor_file = dir / filename
    assert tensor_file.exists(), f"File {tensor_file} does not exist."
    return torch.load(tensor_file)
#####

#### Move?
def dtype_resolve(dtype: torch.dtype | str | None) -> torch.dtype | None:
    """
    Resolve the dtype from a string or return None.
    """
    if isinstance(dtype, str):
        _map = {"float32": torch.float32, "fp32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16}
        dtype_out = _map.get(dtype.lower(), None)
    elif isinstance(dtype, torch.dtype):
        dtype_out = dtype
    else:
        dtype_out = None
    return dtype_out


# A function instead of a class
def run_fold(ds: KinodataDocked,
             gnn_model: RegressionModel,
             config: Config) -> None:
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

    # Buffers for per-batch chunks, then concatenate at end
    layer_bufs: Dict[str, List[torch.Tensor]] = {}
    prior_buf: List[torch.Tensor] = []
    idents: List[int] = []

    dtype_out = dtype_resolve(config.dtype_out)

    gnn_model.eval()
    if config.device:
        device = torch.device(config.device)
    else:
        device = torch.device("cpu")
    gnn_model.to(device)

    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing graph representations"):
            # IDs
            batch_idents = batch.ident.tolist()
            idents.extend(batch_idents)

            # Move to model device
            batch = batch.to(device)

            if not config.graph_level:
                raise NotImplementedError("Only graph-level probing is implemented for now.")
                _ , intermediate_node_reprs, intermediate_edge_reprs, _ = gnn_model(batch)

            # Forward
            _ , intermediate_node_reprs, _ , prior_readout = gnn_model(batch)

            # Pool each layer with the model's aggregator
            # intermediate_node_reprs: {layer_name: (node_repr, batch_index)}
            for layer_name, (node_repr, batch_index) in intermediate_node_reprs.items():
                graph_repr = gnn_model.aggr(node_repr, batch_index).detach().cpu()

                # Possible memory optimization
                if dtype_out is not None:
                    graph_repr = graph_repr.to(dtype_out)

                # Create a buffer for each layer
                if layer_name not in layer_bufs:
                    layer_bufs[layer_name] = []

                layer_bufs[layer_name].append(graph_repr)

            # Same stuff for prior readout once
            pr = prior_readout.detach().cpu()
            if dtype_out is not None:
                pr = pr.to(dtype_out)
            prior_buf.append(pr)
        
        # Concatenate batches per-layer and prior_readout to save per fold
        layers_cat: Dict[str, torch.Tensor] = {}
        for layer_name, chunks in layer_bufs.items():
            layers_cat[layer_name] = torch.cat(chunks, dim=0)  # [N_fold, d]

        prior_cat = torch.cat(prior_buf, dim=0)  # [N_fold, d_pr]
        ids_tensor = torch.tensor(idents, dtype=torch.long)

        # Checks
        assert ids_tensor.shape[0] == fold_size, "IDs tensor size mismatch with fold size"
        assert prior_cat.shape[0] == fold_size, "Prior tensor size mismatch with fold size"
        for layer_name, layer_cat in layers_cat.items():
            assert layer_cat.shape[0] == fold_size, f"Layer {layer_name} tensor size mismatch with fold size"

        # Save to disk separately
        for layer_name, layer_cat in layers_cat.items():
            save_out_tensor(layer_cat, config.output_dir, f"{layer_name}_{config.split_index}.pt", config.split_index)
        save_out_tensor(prior_cat, config.output_dir, f"prior_{config.split_index}.pt", fold = config.split_index)
        save_out_tensor(ids_tensor, config.output_dir, f"ids_{config.split_index}.pt", fold = config.split_index)

    return



def aggregate_fold(config: Config) -> None:
    # TODO: 
    # looping over folds and layers. For each layer, loop over the folds, 
    # go to the fold directory and load the tensors and concatenate them
    # write the concatenated tensor to disk, in the parent of folds
    # access the output directory via config

    # I'm assuming this directory is: 
    output_dir = config.output_dir

    return

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

