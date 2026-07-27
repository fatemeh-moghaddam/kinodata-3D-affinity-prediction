from pathlib import Path
from typing import List, Dict

import torch
from torch_geometric.data import HeteroData

from tqdm import tqdm

from kinodata.data import KinodataDocked
from kinodata.transform import TransformToComplexGraph

from prob.paths_and_io import save_out_tensor





def count_n_complex_no_rdkit(graph: HeteroData, 
                    node_type: str = "ligand") -> int:
    """
    Counts nitrogen atoms (Z == 7) in a HeteroData graph.
    """
    ligand = graph[node_type] if node_type in graph.node_types else None
    if ligand is not None:
        if hasattr(ligand, "z") and isinstance(ligand.z, torch.Tensor):
            return int((ligand.z == 7).sum().item())
    return -1


def count_n_dataset(dataset: List[HeteroData],
                    node_type: str = "ligand",
                    smiles_key: str = "smiles",
                    ) -> List[int]:
    """ 
        Counts nitrogen atoms (Z == 7) in a list of HeteroData graphs.
        Returns a dictionary mapping graph.ident to the number of nitrogen atoms of that complex.
    """
    nitros : Dict[int, int] = {}
    for graph in tqdm(dataset):
        n = count_n_complex_no_rdkit(graph)
        if n != -1:
            nitros[graph.ident.item()] = n
    return nitros


def caculate_save_nitrogen_counts(dataset: List[HeteroData],
                    output_dir: Path,
                    node_type: str = "ligand",
                    smiles_key: str = "smiles",
                    ) -> List[int]:
    """ 
        Counts nitrogen atoms (Z == 7) in a list of HeteroData graphs.
        Saves the result to a file.
    """
    nitrogen_atoms = count_n_dataset(dataset)
    save_out_tensor(nitrogen_atoms, output_dir=output_dir, filename="nitrogen_counts.pt")





if __name__ == "__main__":
    dataset = KinodataDocked(transform=TransformToComplexGraph(remove_heterogeneous_representation=False), use_multiprocessing=True, num_processes= 16)
    caculate_save_nitrogen_counts(dataset)