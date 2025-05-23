import numpy as np

from prob.prob_dataset import ProbingDataset 



def test_get_X_graph_level(graph_list, num_graphs, hidden_channels):
    pd = ProbingDataset.__new__(ProbingDataset)  # bypass __init__ (we only need methods)
    X = pd.get_X(graphs=graph_list, layer_name="layer_0", level="graph")
    assert X.shape == (num_graphs, hidden_channels)          



def test_get_y_mw(graph_list, num_graphs, base_mw, mw_step):
    pd = ProbingDataset.__new__(ProbingDataset)
    y = pd.get_y(graphs=graph_list, target="mw", level="graph")
    
    assert y.shape == (num_graphs,)
    
    mw_range = np.arange(base_mw, base_mw + num_graphs * mw_step, mw_step)
    assert np.allclose(np.unique(y), mw_range)


# TO-DO: add tests for get_y_graph_level and get_y_node_level