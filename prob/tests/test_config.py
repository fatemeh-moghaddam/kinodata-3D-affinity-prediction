from pathlib import Path

# Graph list
NUM_GRAPHS = 2

# GraphReprs object
HIDDEN_CHANNELS = 5
NUM_NODES = 10
NUM_EDGES = 30

# To generate the mw property
BASE_MW = 300.0
MW_STEP = 100.0

# Paths
DATA_ROOT = Path(__file__).parents[3].resolve()
print(f"Data root: {DATA_ROOT}")
# KINO_CKPT_PATH = Path()