#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VAMPNet Training Pipeline

This script provides a command-line interface for training VAMPNet models
on molecular dynamics data.
"""
# Import arguments parser
from args import parse_train_args


import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from datetime import datetime

from pygv.dataset.vampnet_dataset import VAMPNetDataset
from pygv.utils.pipe_utils import find_trajectory_files

from pygv.vampnet import VAMPNet
from pygv.encoder.schnet_wo_embed import SchNetEncoderNoEmbed
from pygv.encoder.meta import Meta

from pygv.scores.vamp_score_v0 import VAMPScore
from pygv.classifier.SoftmaxMLP import SoftmaxMLP

from tqdm import tqdm
from pygv.utils.plotting import plot_vamp_scores

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


def create_dataset_and_loader(args):
    """Create dataset and data loader"""
    # Getting all trajectories in traj directory
    traj_files = find_trajectory_files(args.traj_dir)

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

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available() and not args.cpu
    )

    return dataset, loader


def create_model(args, dataset):
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

    # Create VAMPNet model
    model = VAMPNet(
        encoder=encoder,
        vamp_score=vamp_score,
        classifier_module=classifier,
        lag_time=args.lag_time
    )

    #TODO:FIX
    model.to('cuda')

    return model


def train_model(args, model, loader, paths):
    """Train the model"""
    # Set device
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")

    # Create optimizer
    optimizer = torch.optim.Adam(
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
        data_loader=loader,
        optimizer=optimizer,
        n_epochs=args.epochs,
        device=device,
        save_dir=paths['model_dir'],
        save_every=args.save_every if args.save_every > 0 else None,
        clip_grad_norm=args.clip_grad,
        plot_scores=True,
        plot_path=paths['scores_plot'],
        smoothing=5,
        verbose=True
    )

    return scores


def train_model_new(args, model, loader, paths):
    """Train the model"""
    # Set device
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")

    # Create optimizer
    optimizer = torch.optim.Adam(
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
    #        print(f"  {status} {name}: shape={param.shape}, size={param.numel()}")

    print(f"\nTotal parameters in optimizer: {param_count:,} in {param_tensors} tensors")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train the model
    print(f"Training model on {device}...")

    # Create VAMPScore instance for scoring
    vamp_scorer = VAMPScore(method='VAMP2')

    # Handle batch normalization for small batches
    batch_norm_present = False
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
            batch_norm_present = True
            module.eval()  # Keep BatchNorm layers in eval mode

    if batch_norm_present:
        print("Note: BatchNorm layers are in eval mode to prevent issues with small batches")

    # Create save directory if it doesn't exist
    os.makedirs(paths['model_dir'], exist_ok=True)

    # Training loop
    vamp_scores = []
    best_score = float('-inf')
    best_epoch = 0

    print(f"Starting training for {args.epochs} epochs on {device}")

    for epoch in range(args.epochs):
        #model.train()  # Set model to training mode (except BatchNorm layers)
        epoch_score_sum = 0.0
        n_batches = 0

        # Use tqdm for progress bar
        iterator = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in iterator:
            # Move batch to device
            data_t0, data_t1 = batch
            data_t0 = data_t0.to(device)
            data_t1 = data_t1.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with direct component access - EXACTLY like working script
            encoding_t0, _ = model.encoder(data_t0.x, data_t0.edge_index, data_t0.edge_attr, data_t0.batch)
            encoding_t1, _ = model.encoder(data_t1.x, data_t1.edge_index, data_t1.edge_attr, data_t1.batch)

            chi_t0 = model.classifier_module(encoding_t0)
            chi_t1 = model.classifier_module(encoding_t1)

            # Calculate VAMP loss (negative VAMP score)
            loss = vamp_scorer.loss(chi_t0, chi_t1)

            # Get positive VAMP score for logging
            vamp_score_val = -loss.item()

            # Check for NaN loss
            if torch.isnan(loss).any():
                print(f"Warning: NaN loss detected in epoch {epoch + 1}")
                continue

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping if requested
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

            optimizer.step()

            # Update metrics
            epoch_score_sum += vamp_score_val
            n_batches += 1

        # Calculate average VAMP score for the epoch
        avg_epoch_score = epoch_score_sum / max(1, n_batches)
        vamp_scores.append(avg_epoch_score)

        # Print progress for this epoch
        print(f"Epoch {epoch + 1}/{args.epochs}, VAMP Score: {avg_epoch_score:.4f}")

        # Save best model
        if avg_epoch_score > best_score:
            best_score = avg_epoch_score
            best_epoch = epoch
            if hasattr(model, 'save_complete_model'):
                model.save_complete_model(os.path.join(paths['model_dir'], "best_model.pt"))
            else:
                torch.save(model, os.path.join(paths['model_dir'], "best_model.pt"))
            print("Complete model saved to", os.path.join(paths['model_dir'], "best_model.pt"))

        # Save checkpoint if requested
        if args.save_every and (epoch + 1) % args.save_every == 0:
            if hasattr(model, 'save_complete_model'):
                model.save_complete_model(os.path.join(paths['model_dir'], f"checkpoint_epoch_{epoch + 1}.pt"))
            else:
                torch.save(model, os.path.join(paths['model_dir'], f"checkpoint_epoch_{epoch + 1}.pt"))

    # Save final model
    if hasattr(model, 'save_complete_model'):
        model.save_complete_model(os.path.join(paths['model_dir'], "final_model.pt"))
    else:
        torch.save(model, os.path.join(paths['model_dir'], "final_model.pt"))
    print("Complete model saved to", os.path.join(paths['model_dir'], "final_model.pt"))

    # Plot the VAMP score curve
    plot_vamp_scores(
        scores=vamp_scores,
        save_path=os.path.join(paths['plot_dir'], "vamp_scores.png"),
        smoothing=5,
        title="VAMPNet Training VAMP Scores"
    )
    print("Figure saved to", os.path.join(paths['plots_dir'], "vamp_scores.png"))

    print(f"Training completed. Best VAMP Score: {best_score:.4f} (Epoch {best_epoch + 1})")

    return vamp_scores


def main():
    """Main function"""
    # Parse arguments
    args = parse_train_args()

    # Setup output directory
    paths = setup_output_directory(args)

    # Save configuration
    save_config(args, paths)

    # Create dataset and loader
    dataset, loader = create_dataset_and_loader(args)

    # Create model
    model = create_model(args, dataset)
    print(f"Created VAMPNet model with {sum(p.numel() for p in model.parameters())} parameters")

    # Train model
    scores = train_model(args, model, loader, paths)

    print(f"Training completed successfully. Results saved to {paths['run_dir']}")


if __name__ == "__main__":
    main()
