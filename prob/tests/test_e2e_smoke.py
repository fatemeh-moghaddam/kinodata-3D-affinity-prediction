# from torch_geometric.data import HeteroDataBatch, HeteroData
# from pathlib import Path

# from probing.dataset import ProbingDataset
# from .conftest import dummyKinodataDocked

from kinodata.data import KinodataDocked
from kinodata.model import ComplexTransformer

from prob.utils import build_kd_ds, build_gnn_model
from prob.run_extraction import set_probing_config

import pytest


@pytest.mark.smoke
def test_set_probing_config_default():
    """Test the set_probing_config function."""
    prob_config = set_probing_config()
    
    assert prob_config.gnn_model_type == "CGNN-3D"
    assert prob_config.split_type == "random-k-fold"
    assert prob_config.filter_rmsd_max_value == 2
    assert prob_config.graph_level is True
    assert prob_config.split_index == 0


@pytest.mark.smoke
def test_gnn_model_building(gnn_model):
    """Test the GNN model building."""
    assert gnn_model is not None, "Failed to build GNN model"
    assert isinstance(gnn_model, ComplexTransformer), "GNN model is not of type ComplexTransformer"
    print(gnn_model)


@pytest.mark.smoke
def test_dataset_builds(kd_ds):
    assert len(kd_ds) > 0
    assert isinstance(kd_ds, KinodataDocked)


@pytest.mark.e2e
def test_e2e(prob_config, gnn_model, kd_ds):
    ''' Test the end-to-end pipeline with a smoke test '''


    
