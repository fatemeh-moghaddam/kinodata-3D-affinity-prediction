from prob.generate_prob_ds import set_probing_config



def test_set_probing_config_default():
    """Test the set_probing_config function."""
    prob_config = set_probing_config()
    
    assert prob_config.gnn_model_type == "CGNN-3D"
    assert prob_config.split_type == "random-k-fold"
    assert prob_config.filter_rmsd_max_value == 2
    assert prob_config.graph_level is True
    assert prob_config.split_index == 0


# def test_set_probing_config_custom():
#     """Test the set_probing_config function with custom parameters."""
#     prob_config = set_probing_config(
#         gnn_model_type="CGNN",
#         split_type="scaffold-k-fold",
#         filter_rmsd_max_value=4,
#         graph_level=False,
#         split_index=1
#     )
    
#     assert prob_config.gnn_model_type == "CGNN"
#     assert prob_config.split_type == "scaffold-k-fold"
#     assert prob_config.filter_rmsd_max_value == 4
#     assert prob_config.graph_level is False
#     assert prob_config.split_index == 1