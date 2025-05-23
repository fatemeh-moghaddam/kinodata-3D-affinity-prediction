from prob.prob_dataset import GraphReprs


def test_graph_repr(graph_list, hidden_channels, num_graphs, num_nodes, num_edges, base_mw, mw_step):
    
    assert len(graph_list) == num_graphs, f"Expected {num_graphs} graphs, but got {len(graph_list)}"
    
    # Test the GraphReprs class
    for i, g in enumerate(graph_list):
        assert g.ident == i, f"Expected graph identifier {i}, but got {g.ident}"
        assert g.graph_repr["layer_0"].shape == (hidden_channels,), f"Expected graph_repr shape {(hidden_channels,)}, but got {g.graph_repr['layer_0'].shape}"
        assert g.node_repr["layer_0"].shape == (num_nodes, hidden_channels), f"Expected node_repr shape {(num_nodes, hidden_channels)}, but got {g.node_repr['layer_0'].shape}"
        assert g.edge_repr["layer_0"].shape == (num_edges, hidden_channels), f"Expected edge_repr shape {(num_edges, hidden_channels)}, but got {g.edge_repr['layer_0'].shape}"
        assert g.edge_index["layer_0"].shape == (2, num_nodes), f"Expected edge_index shape {(2, num_nodes)}, but got {g.edge_index['layer_0'].shape}"
        # assert g.node_repr_batch["layer_0"].shape == (num_nodes,), f"Expected node_repr_batch shape {(num_nodes,)}, but got {g.node_repr_batch['layer_0'].shape}"

        if hasattr(g, "mw"):
            assert g.get_property("mw") == base_mw + i * mw_step, f"property mw not equal to {base_mw + i * mw_step}"