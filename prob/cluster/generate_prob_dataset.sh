# Run entirely in HOME; download dataset/models as needed to match utils.py expectations
PROJECT_NAME="kinodata-3D-affinity-prediction"
HOME_PROJ_DIR="${HOME}/${PROJECT_NAME}"
cd "${HOME_PROJ_DIR}"

# Download and extract preprocessed dataset if missing (expects data/processed/...)
if [ ! -d "data/processed" ]; then
    mkdir -p downloads
    wget -O downloads/kinodata3d_processed.zip "https://zenodo.org/records/10886085/files/kinodata3d_processed.zip?download=1" \
    || curl -L -o downloads/kinodata3d_processed.zip "https://zenodo.org/records/10886085/files/kinodata3d_processed.zip?download=1"
    unzip -o downloads/kinodata3d_processed.zip
    rm -f downloads/kinodata3d_processed.zip   # <--- cleanup
fi

# Download and extract pretrained models if missing (expects models/... with .ckpt and config.json)
if [ ! -d "models" ]; then
    mkdir -p downloads
    wget -O downloads/kinodata3d_models.zip "https://zenodo.org/records/10886085/files/kinodata3d_models.zip?download=1" \
    || curl -L -o downloads/kinodata3d_models.zip "https://zenodo.org/records/10886085/files/kinodata3d_models.zip?download=1"
    unzip -o downloads/kinodata3d_models.zip
    rm -f downloads/kinodata3d_models.zip   # <--- cleanup
fi

python3 $HOME_PROJ_DIR/prob/$1.py --split_type $2 --rmsd_cutoff $3 --gnn_model_type $4