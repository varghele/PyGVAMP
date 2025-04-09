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

from pygv.vampnet import VAMPNet
from pygv.encoder.schnet_wo_embed import SchNetEncoderNoEmbed
from pygv.encoder.meta import Meta

from pygv.scores.vamp_score_v0 import VAMPScore
from pygv.classifier.SoftmaxMLP import SoftmaxMLP


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
        for group in args._action_groups:
            f.write(f"# {group.title}\n")
            for action in group._group_actions:
                value = getattr(args, action.dest)
                f.write(f"{action.dest} = {value}\n")
            f.write("\n")


def create_dataset_and_loader(args):
    """Create dataset and data loader"""
    print("Creating dataset...")
    dataset = VAMPNetDataset(
        trajectory_files=args.traj,
        topology_file=args.top,
        lag_time=args.lag_time,
        n_neighbors=args.n_neighbors,
        selection=args.selection,
        stride=args.stride,
        cache_dir=args.cache_dir,
        use_cache=args.cache_dir is not None
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
        classifier = SoftmaxMLP(
            in_channels=args.output_dim,
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
