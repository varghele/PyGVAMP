# PyGVAMP
Please always follow the exact installation steps:
Conda environment creation:
```bash
conda create --name PyGVAMP5 python=3.12
conda activate PyGVAMP5
conda install -c conda-forge -c schrodinger pymol-bundle
pip install matplotlib
pip install joblib
pip install pandas


conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
#Only if conda fails, because with pip there is a bug in 2.5.1
#pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install mdtraj
#On windows, or on pip failure
conda install -c conda-forge mdtraj
```
Path needs to be exported!
export PYTHONPATH=/home/iwe81/PycharmProjects/PyGVAMP:$PYTHONPATH

Install module via
```bash
pip install -e .
pip uninstall pygv
```
