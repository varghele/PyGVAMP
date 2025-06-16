# PyGVAMP

This is the refactoring/new development of [GraphVAMPNets](https://github.com/ghorbanimahdi73/GraphVampNet) from Ghorbani et.al. based on the PyTorch Geometric architecture. Our code achieves a speedup of up to 50x compared to the original, and is built (hopefully) on a more future-proof framework.\
This code however is being actively worked on for our upcoming publications, so be aware, that changes, and pushes to `main` can happen at any moment. A stable release will come at one point, but not yet. 

## 1. Installation/Environment setup
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
```bash
export PYTHONPATH=/home/iwe81/PycharmProjects/PyGVAMP:$PYTHONPATH
```

Install module via
```bash
pip install -e .
pip uninstall pygv
```

## 2. Training the PyGVAMP

We currently supply you with different methods to call the training function, the main functionality being in `pygv.pipe.training.run_training`. You can either call the training directly  with the use of the prepared `run_training.py` in `cluster_scripts`, you can modify the shell script in there if you want to run training on a SLURM cluster, or you can use the `train.py` in `area52`. The current project is in development, and a more high-level, user-friendly training script is in the works.

### 2.1 Calling the training function directly `cluster_scripts/run_training.py`

```bash
# Train VAMPNet model on ATR protein molecular dynamics data
python run_training.py \
    # === Data Configuration ===
    --protein_name "your_protein_name" \                 # Name of the protein 
    --top "path_to_your_topologoy/prot.pdb" \            # Path to topology/structure file
    --traj_dir "path_to_your_trajectories_directory/" \  # Directory containing trajectory files
    --file_pattern "*.xtc" \                             # File pattern for trajectory files
    --selection "name CA" \                              # Atom selection (MDTRAJ based)
    --stride 10 \                                        # Trajectory stride
    --lag_time 20 \                                      # Lag time in nanoseconds
    \
    # === Data Processing ===
    --val_split 0.05 \                                   # Validation data percentage
    --sample_validate_every 100 \                        # Validate every N batches during training
    --use_cache \                                        # Use cached preprocessed data if available
    --cache_dir 'path_to_cache/cache' \                  # Directory to store cached data
    \
    # === Graph Construction ===
    --n_neighbors 20 \                                   # Number of nearest neighbors
    --node_embedding_dim 32 \                            # Dimension for initial node embeddings
    --gaussian_expansion_dim 16 \                        # Dimension for distance feature expansion (edge feats)
    \
    # === SchNet Encoder Architecture ===
    --node_dim 32 \                                      # Node feature dimension
    --edge_dim 16 \                                      # Edge feature dimension  
    --hidden_dim 32 \                                    # Hidden layer dimension
    --output_dim 32 \                                    # Final encoder output dimension
    --n_interactions 4 \                                 # Number of message-passing layers
    --activation 'tanh' \                                # Activation function for encoder
    --use_attention \                                    # Enable attention mechanism
    \
    # === State Classification ===
    --n_states 5 \                                       # Number of states
    --clf_hidden_dim 32 \                                # Classifier hidden dimension
    --clf_num_layers 2 \                                 # Number of classifier layers
    --clf_dropout 0.01 \                                 # Dropout rate for classifier
    --clf_activation 'leaky_relu' \                      # Classifier activation function
    --clf_norm 'LayerNorm' \                             # Normalization type for classifier
    \
    # === Embedding MLP (for atom types) ===
    --use_embedding \                                    # Enable embedding MLP for categorical features
    --embedding_in_dim 42 \                              # Input dimension (number of atoms/residues to be analyzed)
    --embedding_hidden_dim 64 \                          # Hidden dimension for embedding layers
    --embedding_out_dim 32 \                             # Output dimension of embedding
    --embedding_num_layers 2 \                           # Number of embedding MLP layers
    --embedding_dropout 0.01 \                           # Dropout rate for embedding
    --embedding_act 'leaky_relu' \                       # Activation function for embedding
    --embedding_norm 'none' \                            # No normalization for embedding
    \
    # === Training Configuration ===
    --epochs 25 \                                        # Number of training epochs
    --batch_size 128 \                                   # Batch size for training
    --lr 0.001 \                                         # Learning rate
    --weight_decay 1e-5 \                                # L2 regularization strength
    --clip_grad \                                        # Enable gradient clipping
    \
    # === Analysis & Output ===
    --max_tau 200 \                                      # Maximum lag time for timescale analysis
    --output_dir 'path_to_output_dir' \                  # Directory to save results
    --save_every 0 \                                     # Save intermediate checkpoints every N epochs (0=disabled)
    --run_name 'name_of_your_run'                        # Name for training run
```

To see all available arguments of the function, you either look into `pygv.args.args_train.py` or you can simply can call:
```bash
python pygv/args/args_train.py
```

### 2.2 Using the scripts on a SLURM cluster
If you want to run scripts on SLURM cluster, you can adapt the existing shell script `cluster_scripts/atr.sh`. There, you find our code to run the PyGVAMP on a SLURM cluster for the ATR protein. Simply modify the call to the python function, and then run it on the cluster via:
```bash
sbatch -a X-Y atr.sh
```
where X and Y are subsitutes for your number of states you want to run through.

### 2.3 Using the predefined `area52/train.py`
You can simply modify the training script in the `area52` folder. There, simply modify the `create_test_args()` function and paste whatever you need in there. Then, simply execute it via:
```bash
python area52/train.py
```

## 3. Running Analysis
We have included an analysis script that produces the following from the training data:
* pymol plots of state ensembles with and without attention (with .pdb)
* residue attentions (edge) for every state | residue to residue
* residue attentions (edge) for every state | full residue attention
* state transition matrix (with and without self-transitions)
* state network plot with state transitions
* embeddings and state probabilities for each frame of the trajectory

It is easiest to modify the script in `area52/anly.py` for now and then run it via:
```bash
python area52/anly.py
```
YOU ONLY NEED TO MODIFY THE FOLLOWING LINE:
```python
# Base directory of the trained model
base_output_dir = os.path.expanduser('area58/ATR_8_5000_10_v1')
```
During training, implied time scale plots will be done automatically and can be found in the `plots` folder of your output path.