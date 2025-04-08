import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Your imports
from pygv.dataset.VAMPNetDataset import VAMPNetDataset
from pygv.scores.vamp_score_v0 import VAMPScore
from pygv.encoder.schnet_wo_embed import SchNetEncoder
from pygv.vampnet import VAMPNet
from pygv.utils.plotting import plot_vamp_scores


def train_vampnet(dataset_path="testdata", topology_file="topology.pdb"):
    # Find all .xtc files in the dataset directory
    import glob
    import os

    # Find all .xtc files in the dataset directory
    xtc_files = glob.glob(os.path.join(dataset_path, "**/*.xtc"), recursive=True)

    if not xtc_files:
        print(f"No .xtc files found in {dataset_path}")
        return

    print(f"Found {len(xtc_files)} trajectory files")

    # Initialize the dataset
    dataset = VAMPNetDataset(
        trajectory_files=xtc_files,
        topology_file=topology_file,
        lag_time=20,  # Lag time in ns
        n_neighbors=20,  # Number of neighbors for graph construction
        node_embedding_dim=32,
        gaussian_expansion_dim=16,
        selection="name CA",  # Select only C-alpha atoms
        stride=40,  # Take every 2nd frame to reduce dataset size
        cache_dir="testdata",
        use_cache=True
    )

    print(f"Dataset loaded with {len(dataset)} time-lagged pairs")

    # Create data loader
    batch_size = 256  # Small batch size for testing
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the VAMPScore module
    vamp_score = VAMPScore(epsilon=1e-6)

    # Create Meta encoder
    hidden_dim = 64
    output_dim = 16
    num_layers = 4
    embedding_type = "global"  # Use global embeddings for graph-level tasks

    # Initialize the Meta model
    """encoder = Meta(
        node_dim=32,
        edge_dim=16,
        global_dim=64,
        num_node_mlp_layers=2,
        num_edge_mlp_layers=2,
        num_global_mlp_layers=2,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_meta_layers=num_layers,
        embedding_type=embedding_type,
        act="elu",
        norm=None,#"batch_norm",
        dropout=0.0
    )"""

    # Create SchNet encoder
    encoder = SchNetEncoder(
        node_dim=16,#node_dim,
        edge_dim=8,#edge_dim,
        hidden_dim=16,
        output_dim=16,
        n_interactions=3,
        activation='tanh',
        use_attention=True
    )

    # Apply weight initialization
    #encoder.apply(init_weights)

    # Create the VAMPNet model
    vampnet = VAMPNet(encoder=encoder, vamp_score=vamp_score, n_classes=4, lag_time=20)

    # Set up optimizer
    learning_rate = 0.005
    optimizer = torch.optim.AdamW(vampnet.parameters(), lr=learning_rate)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Move model to device
    vampnet = vampnet.to(device)

    # Function to move batch to device
    def to_device(batch, device):
        x_t0, x_t1 = batch
        return (x_t0.to(device), x_t1.to(device))

    # Train for a few epochs to check if it's working
    n_epochs = 100
    print(f"Starting training for {n_epochs} epochs")

    # Train the model
    vamp_scores = []

    for epoch in range(n_epochs):
        epoch_score_sum = 0.0
        n_batches = 0

        with tqdm(loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=True) as t:
            for batch in t:
                # Move batch to device
                data_t0, data_t1 = to_device(batch, device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                # The model expects: forward(self, x, edge_index, edge_attr, batch=None)
                chi_t0, _ = vampnet(data_t0)
                chi_t1, _ = vampnet(data_t1)

                # Calculate VAMP loss (negative VAMP score)
                loss = vamp_score.loss(chi_t0, chi_t1)

                # Get positive VAMP score
                vamp_score_epoch = -loss.item()

                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"Warning: NaN loss detected in epoch {epoch + 1}")
                    continue

                # Backward pass and optimization
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(vampnet.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_score_sum += vamp_score_epoch
                n_batches += 1

            # Calculate average VAMP score for the epoch
            avg_epoch_score = epoch_score_sum / max(1, n_batches)
            vamp_scores.append(avg_epoch_score)
        print(f"Epoch {epoch + 1}/{n_epochs}, VAMP Score: {avg_epoch_score:.4f}")

        # Print progress
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_epoch_score:.4f}")

    # Plot the VAMP score curve
    plot_vamp_scores(
        vamp_scores,
        save_path="vampnet_training_scores.png",
        smoothing=5,  # Apply smoothing with window size 5
        title="VAMPNet Training VAMP Scores"
    )

    # Save the model
    vampnet.save(filepath="mdl_save/mdl_data.pt")

    # Test the model on a sample from the dataset
    print("Testing model on a sample batch...")
    sample_batch = next(iter(loader))
    sample_t0, sample_t1 = to_device(sample_batch, device)

    with torch.no_grad():
        # Get embeddings
        emb_t0, _ = vampnet(sample_t0)
        emb_t1, _ = vampnet(sample_t1)

        # Calculate VAMP score
        score = vamp_score(emb_t0, emb_t1)
        print(f"VAMP score on sample batch: {score.item():.4f}")

    # Check if embeddings look reasonable
    print("\nEmbedding statistics:")
    print(f"T0 embeddings shape: {emb_t0.shape}")
    print(f"T0 embeddings mean: {emb_t0.mean().item():.4f}")
    print(f"T0 embeddings std: {emb_t0.std().item():.4f}")
    print(f"T1 embeddings shape: {emb_t1.shape}")
    print(f"T1 embeddings mean: {emb_t1.mean().item():.4f}")
    print(f"T1 embeddings std: {emb_t1.std().item():.4f}")

    return vampnet, vamp_scores


if __name__ == "__main__":
    # Adjust paths as needed
    model, losses = train_vampnet(
        dataset_path="/home/iwe81/PycharmProjects/DDVAMP/datasets/ab42/trajectories/trajectories/red/",
        #dataset_path="/home/iwe81/PycharmProjects/DDVAMP/datasets/ATR/",
        topology_file="/home/iwe81/PycharmProjects/DDVAMP/datasets/ab42/trajectories/topol.pdb"
        #topology_file = "/home/iwe81/PycharmProjects/DDVAMP/datasets/ATR/prot.pdb",
    )

    # Save the trained model
    torch.save(model.state_dict(), "vampnet_meta_model.pt")
    print("Model saved to vampnet_meta_model.pt")
