PROJECT_NAME="kinodata-3D-affinity-prediction"
HOME_PROJ_DIR="${HOME}/${PROJECT_NAME}"
# CPU_COUNT="64"
cd "${HOME_PROJ_DIR}"
export PYTHONPATH="${HOME_PROJ_DIR}:${PYTHONPATH:-}"

export WANDB_API_KEY=$(cat wandb_api_key)

# Ensure pip stays below 24 to avoid non-standard specifier enforcement warnings
python3 -m pip install --upgrade "pip<24"
# xgboost backs the GPU path of the "random_forest" probe (see prob_models.py).
pip install --upgrade "wandb>=0.15,<1" colorama xgboost

# Keep math backends single-threaded to avoid CPU oversubscription on cluster.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 prob/skyline_targets.py