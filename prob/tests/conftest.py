import torch
import pytest
from kinodata_3D_affinity_prediction.prob.prob_dataset import GraphReprs


@pytest.fixture
def graph_repr():
    # Create a sample GraphReprs object
    # let's assume hideen_channels = 5, num_nodes = 10, num_edges = 30
    g1 = GraphReprs(ident = 1)
    g1.graph_repr = { "layer_0": (torch.randn(5))}
    g1.node_repr = { "layer_0": (torch.randn(10,5))}   
    g1.edge_repr = { "layer_0": (torch.randn(30,5))}	
    g1.edge_index = { "layer_0": (torch.randint(0, 10, (2, 30)))}
    g1.node_repr_batch = { "layer_0": torch.ones(10, dtype=torch.long)}

    g1.add_property("mw", 300.0)
    
    g2 = GraphReprs(ident = 2)	
    g2.graph_repr = { "layer_0": (torch.randn(5))}
    g2.node_repr = { "layer_0": (torch.randn(10,5))}
    g2.edge_repr = { "layer_0": (torch.randn(30,5))}
    g2.edge_index = { "layer_0": (torch.randint(0, 10, (2, 30)))}
    g2.node_repr_batch = { "layer_0": torch.ones(10, dtype=torch.long)*2}
    
    g2.add_property("mw", 400.0)
    return g1