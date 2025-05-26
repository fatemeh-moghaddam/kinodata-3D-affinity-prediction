import torch

from pathlib import Path
from typing import Literal, List, Dict, Any
import json

from .prob_dataset import GraphReprs

# Plan: Saving dicts of {ident: reprs_tensor} per layer per level i.e. per experiment
# needs: making directory structure, saving tensors, loading tensors

Level = Literal["graph", "node", "edge"]

# ─────────────────────────────────────────────────────────────
# Directory setup
# ─────────────────────────────────────────────────────────────
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def init_experiment(root: Path, 
                    exp_id: str, 
                    meta: Dict[str, Any] | None = None
                    ) -> Path:
    """
    Create the experiment directory and write meta.json
        exp_id is made with model+target e.g. cgnn256_rmsd2_mw
    Returns the Path to the experiment root.
    """
    exp_dir = _ensure_dir(root / exp_id)
    if meta:
        with open(exp_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str) # default=str to handle non-JSON-serializable objects, like Path or datetime
    return exp_dir


# ─────────────────────────────────────────────────────────────
# Per layer-level Dict Collection
# ─────────────────────────────────────────────────────────────

def _collect_dict(graphs: List[GraphReprs],
                    level: Level,
                    layer_name: str
                    ) -> Dict[int, torch.Tensor]:
    """
    Collect {ident: tensor} from list of GraphReprs for a layer and level.
    """
    out = {}
    for g in graphs:
        if level == "graph":
            out[g.ident] = g.graph_repr[layer_name]
        elif level == "node":
            out[g.ident] = g.node_repr[layer_name]
        elif level == "edge":
            out[g.ident] = g.edge_repr[layer_name]
        else:
            raise ValueError(f"Invalid level: {level}")
    return out

# ─────────────────────────────────────────────────────────────
# Save/Load Functions
# ─────────────────────────────────────────────────────────────
def save_layer(graphs: List[GraphReprs],
                    level: Level,
                    layer: str,
                    exp_dir: Path
                    ) -> Path:
    """
    Save a single layer-level tensor as a .pt file.
    Example: probing/cgnn256_rmsd2_mw/graph_level/prior_readout.pt
    """
    repr_dict = _collect_dict(graphs, level, layer)
    out_file = _ensure_dir(exp_dir / f"{level}_level") / f"{layer}.pt"
    torch.save(repr_dict, out_file)
    return out_file


def load_layer_dict(level: Level, layer: str, exp_dir: Path
                    ) -> Dict[int, torch.Tensor]:
    """
    Load a layer-level tensor from a .pt file.
    Example: probing/cgnn256_rmsd2_mw/graph_level/prior_readout.pt
    """
    in_file = exp_dir / f"{level}_level" / f"{layer}.pt"
    if not in_file.exists():
        raise FileNotFoundError(f"File {in_file} does not exist")
    repr_dict = torch.load(in_file)
    return repr_dict


def load_to_GraphReprs(
                        repr_dict: Dict[int, torch.Tensor],
                        layer: str,
                        level: Level
                    ) -> List[GraphReprs]:
    """
    Convert a layer-level dict {ident: tensor} to a list of GraphReprs.
    """
    graphs = []
    for ident, tensor in repr_dict.items():
        g = GraphReprs(ident=ident)
        if level == "graph":
            g.graph_repr[layer] = tensor
        elif level == "node":
            g.node_repr[layer] = tensor
        elif level == "edge":
            g.edge_repr[layer] = tensor
        else:
            raise ValueError(f"Invalid level: {level}")
        graphs.append(g)
    return graphs