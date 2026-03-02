from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

_ROOT = Path(os.environ.get("HOME_PROJ_DIR", Path(__file__).resolve().parents[2]))
_INTERACTIONS_DIR = _ROOT / "data" / "plip_kinodata3d" / "processed"


@lru_cache
def load_ident_activity_mapping(
    mappings: Optional[dict[str, str]] = {"activities.activity_id": "activity_id",
                                          "smiles_processed" : "smiles"}
    ) -> pd.DataFrame:
    df = pd.read_csv(_ROOT / "data" / "ident_to_activity_id.csv")
    df = df.rename(columns=mappings)
    return df


@lru_cache
def load_interaction(name: str) -> pd.DataFrame:
    """
    name in {"hydrogen_bonds", "halogen_bonds", "hydrophobic_interactions", ...}
    """
    path = _INTERACTIONS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def attach_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach activity_id to an interaction DF via ident_to_activity_id mapping.
    Assumes interaction DF has an 'activity_id' column or something equivalent.
    """
    mapping = load_ident_activity_mapping()
    return df.merge(mapping, on="activity_id", how="left")