import torch
import pytest
from prob.prob_dataset import GraphReprs
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


@pytest.fixture
def graph_list(num_graphs, num_nodes, num_edges, hidden_channels, base_mw, mw_step):
    # Create a list of GraphReprs object

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
    
