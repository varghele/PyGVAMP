#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VAMPNet Training Pipeline

This script provides a command-line interface for training VAMPNet models
on molecular dynamics data.
"""
from pymol.querying import distance

# Import arguments parser
from pygv.args import parse_train_args


import os
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from datetime import datetime

from pygv.dataset.vampnet_dataset import VAMPNetDataset
from pygv.utils.pipe_utils import find_trajectory_files
from pygv.utils.analysis import analyze_vampnet_outputs
from pygv.utils.ck import run_ck_analysis
from pygv.utils.its import analyze_implied_timescales
from pygv.utils.nn_utils import init_for_vamp

from pygv.vampnet import VAMPNet
from pygv.encoder.schnet_wo_embed_v2 import SchNetEncoderNoEmbed
from pygv.encoder.meta_att import Meta

from pygv.scores.vamp_score_v0 import VAMPScore
from pygv.classifier.SoftmaxMLP import SoftmaxMLP

from torch_geometric.nn.models import MLP


def setup_output_directory(args):
    """Setup output directory and return paths"""
    # Create run name if not provided
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    run_dir = os.path.join(args.output_dir, args.run_name)
    model_dir = os.path.join(run_dir, "models")
    plot_dir = os.path.join(run_dir, "plots")

    for directory in [run_dir, model_dir, plot_dir]:
        os.makedirs(directory, exist_ok=True)

    # Create paths
    paths = {
        'run_dir': run_dir,
        'model_dir': model_dir,
        'plot_dir': plot_dir,
        'config': os.path.join(run_dir, 'config.txt'),
        'scores_plot': os.path.join(plot_dir, 'vamp_scores.png'),
        'best_model': os.path.join(model_dir, 'best_model.pt'),
        'final_model': os.path.join(model_dir, 'final_model.pt'),
    }

    return paths


def save_config(args, paths):
    """Save configuration to a text file"""
    with open(paths['config'], 'w') as f:
        f.write("# Configuration\n\n")

        # Convert args to a dictionary
        args_dict = vars(args)

        # Write each key-value pair
        for key, value in sorted(args_dict.items()):
            f.write(f"{key} = {value}\n")


def create_dataset_and_loader(args,
                              is_frame_loader=False,
                              test_split: float = 0.2,
                              seed: int = 42):
    """
    Create dataset and data loaders for training and testing.

    Parameters
    ----------
    args : Namespace
        Arguments containing dataset and loader configurations.
    is_frame_loader : bool, default=False
        Whether to create a frame-wise dataset loader (for inference).
    test_split : float, default=0.2
        Fraction of the dataset to use for testing.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (dataset, train_dataset, train_loader, test_dataset, test_loader)
    """
    # Getting all trajectories in traj directory
    traj_files = find_trajectory_files(args.traj_dir, file_pattern=args.file_pattern)

    print("Creating dataset...")
    dataset = VAMPNetDataset(
        trajectory_files=traj_files,
        topology_file=args.top,
        lag_time=args.lag_time,
        n_neighbors=args.n_neighbors,
        node_embedding_dim=args.node_embedding_dim,
        gaussian_expansion_dim=args.gaussian_expansion_dim,
        selection=args.selection,
        stride=args.stride,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache
    )

    print(f"Dataset created with {len(dataset)} samples")

    # Split dataset into training and testing sets
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size

    # Set random seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    print(f"Training set: {len(train_dataset)} samples")
    print(f"Testing set: {len(test_dataset)} samples")

    # If individual frames are needed (for the tests), return a framewise dataset
    if is_frame_loader:
        # Get frames dataset instead of time-lagged pairs dataset
        frames_dataset = dataset.get_frames_dataset(return_pairs=False)

        train_loader = DataLoader(
            frames_dataset,
            shuffle=False,  # Always false for inference
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available() and not args.cpu
        )
        test_loader = None  # Frame loader is typically used for inference only
    else:
        # Create data loaders for training and testing
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available() and not args.cpu
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available() and not args.cpu
        )

    return dataset, train_dataset, train_loader, test_dataset, test_loader


def create_model(args):
    """Create VAMPNet model"""
    # Create encoder based on selected type
    if args.encoder_type.lower() == 'schnet':
        encoder = SchNetEncoderNoEmbed(
            node_dim=args.node_dim,
            edge_dim=args.edge_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            n_interactions=args.n_interactions,
            activation=args.activation,
            use_attention=args.use_attention
        )
    elif args.encoder_type.lower() == 'meta':
        encoder = Meta(
            node_dim=args.meta_node_dim,
            edge_dim=args.meta_edge_dim,
            global_dim=args.meta_global_dim,
            num_node_mlp_layers=args.meta_num_node_mlp_layers,
            num_edge_mlp_layers=args.meta_num_edge_mlp_layers,
            num_global_mlp_layers=args.meta_num_global_mlp_layers,
            hidden_dim=args.meta_hidden_dim,
            output_dim=args.meta_output_dim,
            num_meta_layers=args.meta_num_meta_layers,
            embedding_type=args.meta_embedding_type,
            act=args.meta_activation,
            norm=args.meta_norm,
            dropout=args.meta_dropout
        )
    elif args.encoder_type.lower() == 'ml3':
        encoder = None
        #TODO: IMPLEMENT
        """encoder = ML3Encoder(
            node_dim=args.ml3_node_dim,
            edge_dim=args.ml3_edge_dim,
            hidden_dim=args.ml3_hidden_dim,
            output_dim=args.ml3_output_dim,
            num_layers=args.ml3_num_layers,
            activation=args.ml3_activation
        )"""
    else:
        raise ValueError(f"Unsupported encoder type: {args.encoder_type}. "
                         f"Choose from 'schnet', 'meta', or 'ml3'.")

    # Get output dimension based on encoder type
    if args.encoder_type.lower() == 'schnet':
        output_dim = args.output_dim
    elif args.encoder_type.lower() == 'meta':
        output_dim = args.meta_output_dim
    elif args.encoder_type.lower() == 'ml3':
        output_dim = args.ml3_output_dim

    # Create VAMP score module
    vamp_score = VAMPScore(epsilon=1e-6, mode='regularize')

    # Create classifier if requested
    classifier = None
    if args.n_states > 0:
        if args.encoder_type == 'schnet':
            classifier = SoftmaxMLP(
                in_channels=args.output_dim,
                hidden_channels=args.clf_hidden_dim,
                out_channels=args.n_states,
                num_layers=args.clf_num_layers,
                dropout=args.clf_dropout,
                act=args.clf_activation,
                norm=args.clf_norm
            )
        elif args.encoder_type == 'meta':
            classifier = SoftmaxMLP(
                in_channels=args.meta_output_dim,
                hidden_channels=args.clf_hidden_dim,
                out_channels=args.n_states,
                num_layers=args.clf_num_layers,
                dropout=args.clf_dropout,
                act=args.clf_activation,
                norm=args.clf_norm
            )

    if args.use_embedding:
        # Create an embedding module
        embedding_module = MLP(
                    in_channels=args.embedding_in_dim,
                    hidden_channels=args.embedding_hidden_dim,
                    out_channels=args.embedding_out_dim,
                    num_layers=args.embedding_num_layers,
                    dropout=args.embedding_dropout,
                    act=args.embedding_act,
                    norm=args.embedding_norm
                )
    else:
        embedding_module = None

    # Create VAMPNet model
    model = VAMPNet(
        embedding_module=embedding_module,
        encoder=encoder,
        vamp_score=vamp_score,
        classifier_module=classifier,
        lag_time=args.lag_time
    )

    # Apply to your model
    init_for_vamp(model, method='kaiming_normal')
    model.to('cuda')

    return model


def train_model(args, model, train_loader, test_loader, paths):
    """Train the model"""
    # Set device
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Print optimizer parameters
    print("\nParameters captured by optimizer:")
    param_count = 0
    param_tensors = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if this parameter is in the optimizer
            in_optimizer = False
            for group in optimizer.param_groups:
                for opt_param in group['params']:
                    if opt_param is param:
                        in_optimizer = True
                        break
                if in_optimizer:
                    break

            status = "✓" if in_optimizer else "✗"
            param_count += param.numel() if in_optimizer else 0
            param_tensors += 1 if in_optimizer else 0
            print(f"  {status} {name}: shape={param.shape}, size={param.numel()}")

    print(f"\nTotal parameters in optimizer: {param_count:,} in {param_tensors} tensors")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")



    # Train the model
    print(f"Training model on {device}...")
    scores = model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        n_epochs=args.epochs,
        device=device,
        learning_rate=args.lr,
        save_dir=paths['model_dir'],
        save_every=args.save_every if args.save_every > 0 else None,
        clip_grad_norm=args.clip_grad,
        plot_scores=True,
        plot_path=paths['scores_plot'],
        smoothing=5,
        verbose=True,
        show_batch_vamp=True,
        check_grad_stats=False,
        sample_validate_every=args.sample_validate_every
    )

    return scores


def run_training(args):
    """Main function"""
    # Setup output directory
    paths = setup_output_directory(args)

    # Save configuration
    save_config(args, paths)

    # Create dataset and loader
    dataset, train_dataset, train_loader, test_dataset, test_loader = create_dataset_and_loader(args,test_split=args.val_split)

    # Infer num atoms for later embedding
    args.embedding_in_dim = dataset.n_atoms

    # Create model
    model = create_model(args)
    print(f"Created VAMPNet model with {sum(p.numel() for p in model.parameters())} parameters")

    # Train model
    scores = train_model(args=args,
                         model=model,
                         train_loader=train_loader,
                         test_loader=test_loader,
                         paths=paths)

    print(f"Training completed successfully. Results saved to {paths['run_dir']}")

    # Determine device
    if args.cpu is True:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate frames dataset and loader
    # Here we do not need to return pairs, we need to analyze each frame,
    # which is why we need to modify the dataloader a bit to return batches of frames
    dataset, train_dataset, train_loader, _, _ = create_dataset_and_loader(args, is_frame_loader=True)

    # TODO: pre-analysis for CK Tests and ITS Plots
    # Get state transition probabilities, graph embeddings and attention scores
    probs, embeddings, attentions, edge_indices = analyze_vampnet_outputs(model=model,
                                                                          data_loader=train_loader,
                                                                          save_folder=paths['model_dir'],
                                                                          batch_size=args.batch_size if hasattr(args,
                                                                                                                'batch_size') else 32,
                                                                          device=device,
                                                                          return_tensors=False
                                                                          )

    # Get actual timestep of the trajectory
    inferred_timestep = dataset._infer_timestep() / 1000  # Timestep in nanoseconds

    # Get lag times up to maximum lag time
    if args.max_tau is not None:
        # Calculate step size to get between 50 and 200 lag times
        step = max(1, args.max_tau // 200)  # Ensure at least 200 values
        lag_times_ns = list(range(1, args.max_tau, step))

        # If we have fewer than 50 values, reduce step size
        if len(lag_times_ns) < 50:
            step = max(1, args.max_tau // 50)
            lag_times_ns = list(range(1, args.max_tau, step))

        # Ensure we don't exceed 200 values
        if len(lag_times_ns) > 200:
            lag_times_ns = lag_times_ns[:200]

    else:
        max_tau = 250
        # Calculate step size to get between 50 and 200 lag times
        step = max(1, max_tau // 200)  # Ensure at least 200 values
        lag_times_ns = list(range(1, max_tau, step))

        # If we have fewer than 50 values, reduce step size
        if len(lag_times_ns) < 50:
            step = max(1, max_tau // 50)
            lag_times_ns = list(range(1, max_tau, step))

        # Ensure we don't exceed 200 values
        if len(lag_times_ns) > 200:
            lag_times_ns = lag_times_ns[:200]

    # Run Chapman Kolmogorow Test
    run_ck_analysis(
        probs=probs,
        save_dir=paths['plot_dir'],
        protein_name=args.protein_name,
        lag_times_ns=[args.lag_time],
        steps=10,
        stride=args.stride,
        timestep=inferred_timestep
    )
    print("CK test complete")

    # Calculate and plot implied timescales
    analyze_implied_timescales(
        probs=probs,
        save_dir=paths['plot_dir'],
        protein_name=args.protein_name,
        lag_times_ns=lag_times_ns,
        stride=args.stride,
        timestep=inferred_timestep,
    )
    print("ITS calculation complete")






if __name__ == "__main__":
    # Parse arguments
    args = parse_train_args()

    run_training(args)
