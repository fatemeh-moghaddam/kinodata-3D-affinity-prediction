import json
from pathlib import Path

import pytest
import torch


from prob.io import init_experiment, _collect_dict, save_layer, load_layer_dict, load_to_GraphReprs
from prob.prob_dataset import GraphReprs


def test_init_experiment(tmp_path: Path):
    """Test the init_experiment function."""
    model_name = "dummy_model"
    target_name = "dummy_target"
    exp_id = model_name + "_" + target_name
    meta = {"model": model_name, "target": target_name, "description": "This is a test experiment."}
    
    exp_dir = init_experiment(tmp_path, exp_id, meta)
    
    assert exp_dir.exists(), "Experiment directory was not created."
    assert (exp_dir / "meta.json").exists(), "Meta file was not created."
    
    with open(exp_dir / "meta.json", "r") as f:
        loaded_meta = json.load(f)
    
    assert loaded_meta == meta, "Meta data does not match."


def test_collect_dict(graph_list: list[GraphReprs]):
    """Test the _collect_dict function."""
    level = "graph"
    layer = "layer_0"
    
    collected = _collect_dict(graphs=graph_list, level=level, layer=layer)
    
    assert isinstance(collected, dict), "Output should be a dictionary."
    assert len(collected) == len(graph_list), "Output dictionary length does not match number of graphs."
    assert set(collected.keys()) == {g.ident for g in graph_list}, "Keys in output dictionary do not match graph identifiers."

    for g in graph_list:
        assert g.ident in collected, f"Graph {g.ident} not found in collected dictionary."
        if level == "graph":
            assert collected[g.ident].shape == g.graph_repr["layer_0"].shape, "Shape of collected tensor does not match graph representation shape."
        elif level == "node":
            assert collected[g.ident].shape == g.node_repr["layer_0"].shape, "Shape of collected tensor does not match node representation shape."
        elif level == "edge":
            assert collected[g.ident].shape == g.edge_repr["layer_0"].shape, "Shape of collected tensor does not match edge representation shape."


def test_save_load_layer(graph_list: list[GraphReprs], tmp_path: Path):
    """Test the save_layer and load_layer_dict functions."""
    level = "graph"
    layer = "layer_0"
    
    exp_dir = init_experiment(tmp_path, "io_test")
    
    # Save the layer
    saved_file = save_layer(graphs=graph_list, level=level, layer=layer, exp_dir=exp_dir)
    
    assert saved_file.exists(), "Saved file does not exist."
    
    # Load the layer
    loaded_dict = load_layer_dict(exp_dir=exp_dir, level=level, layer=layer)
    
    assert isinstance(loaded_dict, dict), "Loaded data should be a dictionary."
    assert len(loaded_dict) == len(graph_list), "Loaded dictionary length does not match number of graphs."
    
    for g in graph_list:
        assert g.ident in loaded_dict, f"Graph {g.ident} not found in loaded dictionary."
        assert torch.equal(loaded_dict[g.ident], g.graph_repr[layer]), f"Loaded tensor for graph {g.ident} does not match saved tensor."



def test_load_to_GraphReprs(graph_list: list[GraphReprs], tmp_path: Path):
    """Test the load_to_GraphReprs function."""
    level = "graph"
    layer = "layer_0"
    
    exp_dir = init_experiment(tmp_path, "io_test")
    
    # Save the layer
    save_layer(graphs=graph_list, level=level, layer=layer, exp_dir=exp_dir)
    
    # Load the layer dictionary
    repr_dict = load_layer_dict(exp_dir=exp_dir, level=level, layer=layer)
    assert isinstance(repr_dict, dict), "Loaded data should be a dictionary."
    # Load to GraphReprs
    loaded_graphs = load_to_GraphReprs(repr_dict=repr_dict, level=level, layer=layer)
    
    assert isinstance(loaded_graphs, list), "Loaded data should be a list of GraphReprs."
    assert len(loaded_graphs) == len(graph_list), "Loaded graphs length does not match original graphs length."
    
    for g in graph_list:
        loaded_g = next((lg for lg in loaded_graphs if lg.ident == g.ident), None)
        assert loaded_g is not None, f"Graph {g.ident} not found in loaded graphs."
        assert torch.equal(loaded_g.graph_repr[layer], g.graph_repr[layer]), f"Graph {g.ident} representation does not match."