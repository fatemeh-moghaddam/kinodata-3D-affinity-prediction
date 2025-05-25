from kinodata.data import KinodataDocked
from torch_geometric.data import HeteroDataBatch, HeteroData


def test_e2e_smoke():
    ''' Test the end-to-end pipeline with a smoke test '''
    # Create a KinodataDocked object
    orig_data = KinodataDocked()
    
    # Check if the object is created successfully
    assert orig_data is not None, "KinodataDocked object is None"
    assert isinstance(orig_data, HeteroData), "KinodataDocked object is not of the expected type"
    assert len(orig_data) == 41238, "KinodataDocked object does not have the expected number of graphs"
    
    # Check if the object has the expected properties
    assert hasattr(orig_data, "data"), "KinodataDocked object has no data attribute"
    assert hasattr(k, "graph_repr"), "KinodataDocked object has no graph_repr attribute"
    
    # Check if the data is in the expected format
    assert isinstance(orig_data.data, HeteroDataBatch), "KinodataDocked data is not a dictionary"