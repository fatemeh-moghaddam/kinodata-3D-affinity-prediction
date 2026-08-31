
PROJECT_NAME="kinodata-3D-affinity-prediction"
HOME_PROJ_DIR="${HOME}/${PROJECT_NAME}"
cd "${HOME_PROJ_DIR}"
export PYTHONPATH="${HOME_PROJ_DIR}:${PYTHONPATH:-}"

# Disable W&B globally to avoid init inside model code
# export WANDB_DISABLED=true
# export WANDB_MODE=disabled
# Enable W&B for this script
export WANDB_API_KEY=$(cat wandb_api_key)

# Ensure pip stays below 24 to avoid non-standard specifier enforcement warnings
python3 -m pip install --upgrade "pip<24"
pip install --upgrade "wandb>=0.15,<1" colorama
# Run from the job's working directory; assume required code is transferred with the job
# Reduces fragmentation; the extraction peak is a few large short-lived tensors.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -c "import torch; print('cuda available:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

# $5 = overwrite (1/0): recompute folds whose artifacts are already on disk instead of
# skipping them. Defaults to 0 so queue lines that omit it keep the resume behavior.
OVERWRITE="${5:-0}"

python3 prob/$1.py --split_type "$2" --filter_rmsd_max_value "$3" --gnn_model_type "$4" \
    --device cuda --save_representations 1 --save_predictions 1 --include_val 0 --overwrite "${OVERWRITE}"
