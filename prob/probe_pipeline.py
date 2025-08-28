from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, List, Union
import torch
from torch_geometric.loader import DataLoader

# --- import their config module ---
# assuming your config lives at kinodata/config.py (adapt if different)
from kinodata.configuration import register, get, Config

# =============== 1) add a tiny probe section to their config ===============
register("probing", 
        graph_level = True,
        gnn_model = "CGNN-3D",
        split_type = "random-k-fold",
        rmsd_cutoff = 2,
        )

# =============== 2) tiny helpers ===============
def as_list(x: Union[None, str, float, Iterable]):
    if x is None:
        return []
    if isinstance(x, (str, float, int)):
        return [x]
    return list(x)

def setting_key(model_type: str, split_type: str, rmsd: Union[int, float]) -> str:
    return f"{model_type}-{split_type}-{rmsd}"

# =============== 3) dataset & model builders (plug your own) ===============
def build_dataset(cfg: Config) -> "torch.utils.data.Dataset":
    """
    TODO: Replace with your actual KinodataDocked init that respects:
      - cfg.filter_rmsd_max_value
      - cfg.split_type
      - cfg.split_index (current fold)
    """
    # Example sketch (adapt to real API):
    # from kinodata.data import KinodataDocked
    # ds = KinodataDocked(
    #     root=Path("data"),
    #     split_type=cfg.split_type,
    #     split_index=cfg.split_index,
    #     filter_rmsd_max_value=cfg.filter_rmsd_max_value,
    # )
    # return ds
    raise NotImplementedError("Fill build_dataset() with your KinodataDocked construction.")

def build_model(cfg: Config) -> "torch.nn.Module":
    """
    TODO: Load your ComplexTransformer (or other) and its fold-specific checkpoint.
    Must return a model whose forward returns:
      out_graphs, node_dict, edge_dict, prior_readout
    and has .aggr for pooling node -> graph.
    """
    # Example sketch (adapt to real API):
    # from kinodata.model import ComplexTransformer
    # model = ComplexTransformer(**whatever_from_cfg)
    # ckpt = (
    #     Path(cfg.probe_ckpt_dir)
    #     / cfg.probe_model_type
    #     / cfg.split_type
    #     / str(cfg.filter_rmsd_max_value)
    #     / f"fold{cfg.split_index}.ckpt"
    # )
    # state = torch.load(ckpt, map_location="cpu")["state_dict"]
    # model.load_state_dict(state)
    # model.eval()
    # return model
    raise NotImplementedError("Fill build_model() to construct & load your model/ckpt.")

# =============== 4) one inference step that extracts reps ===============
def probe_step(model, batch, want_graph: bool):
    """
    Returns a dict:
      {"graph": {layer: [B,H]},
       "node":  {layer: (X, batch_vec)},
       "edge":  {layer: (E, edge_index)}}
    Only fills what you need (graph vs node/edge) per cfg.probe_graph_level.
    """
    out_graphs, node_dict, edge_dict, prior = model(batch)
    out = {"graph": {}, "node": {}, "edge": {}}
    if want_graph:
        for layer, (x, b) in node_dict.items():
            out["graph"][layer] = model.aggr(x, b)  # pool per-graph
        out["graph"]["prior_readout"] = prior
    else:
        out["node"] = node_dict
        out["edge"] = edge_dict
    return out

# =============== 5) write shards immediately (no RAM build-up) ===============
def write_shard(out_root: Path, setting: str, fold: int, layer: str, payload, shard_id: int, kind: str):
    path = out_root / setting / f"fold{fold}" / kind
    path.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path / f"{layer}-shard{shard_id}.pt")

# =============== 6) run a single fold ===============
def run_one_fold(cfg: Config, out_root: Path, setting: str, device: str):
    # To Do
    ...

# =============== 7) aggregate shards across folds per layer ===============
def aggregate_all_folds(cfg: Config, out_root: Path, setting: str):
    # To Do
    ...

# =============== 8) run a single setting (loops all folds) ===============
def run_one_setting(cfg: Config):
    # To Do
    ...

    aggregate_all_folds(cfg, out_root, setting)

