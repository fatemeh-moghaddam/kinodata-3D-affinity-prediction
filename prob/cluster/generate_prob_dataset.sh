
PROJECT_NAME="kinodata-3D-affinity-prediction"
HOME_PROJ_DIR="${HOME}/${PROJECT_NAME}"
cd "${HOME_PROJ_DIR}"
export PYTHONPATH="${HOME_PROJ_DIR}:${PYTHONPATH:-}"

# Ensure pip stays below 24 to avoid non-standard specifier enforcement warnings
python3 -m pip install --upgrade "pip<24"
pip install colorama
# Run from the job's working directory; assume required code is transferred with the job
python3 prob/$1.py --split_type "$2" --filter_rmsd_max_value "$3" --gnn_model_type "$4"