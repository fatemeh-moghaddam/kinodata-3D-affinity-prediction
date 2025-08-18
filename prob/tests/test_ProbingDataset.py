import pytest
from prob.prob_dataset import ProbingDataset





def test_probing_dataset_init(probing_dataset, loaded_cgnn_model):
    ''' Test the initialization of ProbingDataset '''
    assert probing_dataset is not None, "ProbingDataset object is None"
    assert isinstance(probing_dataset, ProbingDataset), "ProbingDataset object is not of the expected type"
    assert len(probing_dataset) == 10, "ProbingDataset does not have the expected number of graphs"
    assert probing_dataset.graph_level is True, "ProbingDataset graph_level should be True"
    assert probing_dataset.batch_size == 1, "ProbingDataset batch_size should be 1"
    assert probing_dataset.loaded_gnn_model is not None, "GNN model is not loaded in ProbingDataset"
    assert probing_dataset.loaded_gnn_model == loaded_cgnn_model, "Loaded GNN model does not match the expected model"
    



# This is to test the compute_reprs with different batch sizes, 1 and 10
@pytest.mark.parametrize("batch_size", [1, 10])
def test_compute_reprs_batch_size(kino_dataset, batch_size):
    ''' Test the compute_reprs method with different batch sizes '''
    ...