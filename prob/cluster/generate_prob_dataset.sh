# Run entirely in HOME; download dataset/models as needed to match utils.py expectations
PROJECT_NAME="kinodata-3D-affinity-prediction"
HOME_PROJ_DIR="${HOME}/${PROJECT_NAME}"
cd "${HOME_PROJ_DIR}"


python3 $HOME_PROJ_DIR/prob/$1.py --split_type $2 --rmsd_cutoff $3 --gnn_model_type $4