''' Fixtures: GraphReprs, KinodataDocked, CGNN model, model config, and model ckpt.'''
import torch
import pytest
from pathlib import Path
import os

from kinodata.data import KinodataDocked
from kinodata.transform import TransformToComplexGraph
from kinodata.model.complex_transformer import make_model, ComplexTransformer

from prob.utils import load_config, load_model_from_checkpoint
from prob.prob_dataset import GraphReprs, ProbingDataset
from .test_config import *



@pytest.fixture
def hidden_channels():
    return HIDDEN_CHANNELS

@pytest.fixture
def num_graphs():
    return NUM_GRAPHS

@pytest.fixture
def num_nodes():
    return NUM_NODES

@pytest.fixture
def num_edges():
    return NUM_EDGES

@pytest.fixture
def base_mw():
    return BASE_MW

@pytest.fixture
def mw_step():
    return MW_STEP




    

####### The Legacy Fixtures #######

@pytest.mark.legacy
@pytest.fixture(scope="session") # to avoid reloading the dataset multiple times, cause KinodataDocked is large
def kino_dataset():
    ''' Create a subset of KinodataDocked '''
    ds = KinodataDocked(transform=TransformToComplexGraph, use_multiprocessing=True, num_processes=12)
    # limit to first 10 samples
    ds = torch.utils.data.Subset(ds, range(20))
    assert len(ds) == 20, "KinodataDocked dataset should have 20 samples"
    return ds

@pytest.mark.legacy
@pytest.fixture
def model_path():
    ''' Return the path to the model directory '''
    p = Path(__file__).parents[2] / "models" / "rmsd_cutoff_2" / "scaffold-k-fold" / "0" / "CGNN-3D"
    assert p.exists(), f"Model path {p} does not exist"
    return p

@pytest.mark.legacy
@pytest.fixture
def model_config_path(model_path):
    ''' Return the path to the model config file '''
    cfg_path = model_path / "config.json"
    assert cfg_path.exists(), f"Model config path {cfg_path} does not exist"
    return model_path / "config.json"

@pytest.mark.legacy
@pytest.fixture
def model_ckpt(model_path):
    ''' Return the path to the model checkpoint file '''
    ckpt = list(model_path.glob("**/*.ckpt"))
    assert len(ckpt) == 1, "Expected exactly one checkpoint file"
    return ckpt[0]

@pytest.mark.legacy
@pytest.fixture
def cgnn_model(model_config_path):
    ''' Load the model from the config and checkpoint files '''
    config = load_config(model_config_path)
    model = make_model(config)
    # To DO : Add more specific assertions based on the expected model structure
    assert isinstance(model, ComplexTransformer), "Model is not of type ComplexTransformer"
    return model

@pytest.mark.legacy
# I actually don't need this fixture, since the ProbingDataset already loads the model
@pytest.fixture(scope="session")
def loaded_cgnn_model(cgnn_model, model_ckpt):
    ''' Load the CGNN model from the config and checkpoint files '''
    model = load_model_from_checkpoint(cgnn_model, model_ckpt)
    return model

@pytest.mark.legacy
@pytest.fixture(scope="session")
def probing_dataset(kino_dataset, cgnn_model, model_ckpt):
    ''' Create a probing dataset with 10 KinodataDocked objects '''
    pd = ProbingDataset(
        orig_dataset=kino_dataset,
        model=cgnn_model,
        model_ckpt=str(model_ckpt),
        graph_level=True,  # Use graph level representations
        batch_size=1,  
        num_workers=1,
    )
    return pd

@pytest.mark.legacy
@pytest.fixture
def graph_list(num_graphs, num_nodes, num_edges, hidden_channels, base_mw, mw_step):
    ''' Create a list of GraphReprs object '''

    graphs = []
    for i in range(num_graphs):
        g = GraphReprs(ident = i)
        g.graph_repr = { "layer_0": (torch.randn(hidden_channels))}
        g.node_repr = { "layer_0": (torch.randn(num_nodes,hidden_channels))}
        g.edge_repr = { "layer_0": (torch.randn(num_edges,hidden_channels))}
        g.edge_index = { "layer_0": (torch.randint(0, num_nodes, (2, num_nodes)))}
        # g.node_repr_batch = { "layer_0": torch.ones(num_nodes, dtype=torch.long)*i}
        # prob target
        g.add_property("mw", base_mw + i*mw_step)
        graphs.append(g)

    return graphs
