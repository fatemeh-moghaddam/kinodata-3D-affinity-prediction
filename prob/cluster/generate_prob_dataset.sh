
PROJECT_NAME="kinodata-3D-affinity-prediction"
HOME_PROJ_DIR="${HOME}/${PROJECT_NAME}"
cd "${HOME_PROJ_DIR}"
export PYTHONPATH="${HOME_PROJ_DIR}:${PYTHONPATH:-}"

pip install -e .
pip instal colorama
# Run from the job's working directory; assume required code is transferred with the job
python3 prob/$1.py --split_type "$2" --rmsd_cutoff "$3" --gnn_model_type "$4"