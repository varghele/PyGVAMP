#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VAMPNet Training Pipeline

This script provides a command-line interface for training VAMPNet models
on molecular dynamics data.
"""
# Import arguments parser
from pygv.args import parse_train_args


import os
import torch
from torch_geometric.loader import DataLoader
from datetime import datetime

from pygv.dataset.vampnet_dataset import VAMPNetDataset
from pygv.utils.pipe_utils import find_trajectory_files
from pygv.utils.analysis import analyze_vampnet_outputs
from pygv.utils.ck import run_ck_analysis

from pygv.vampnet import VAMPNet
from pygv.encoder.schnet_wo_embed import SchNetEncoderNoEmbed
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
        shuffle=True,
        batch_size=args.batch_size,
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

    #TODO:FIX
    def init_for_vamp(model):
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                # Initialize with slightly larger weights
                torch.nn.init.xavier_uniform_(m.weight, gain=1.5)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.1)  # Small positive bias

    # Apply to your model
    init_for_vamp(model)
    model.to('cuda')

    return model


def train_model(args, model, loader, paths):
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
        data_loader=loader,
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
        verbose=True
    )

    return scores


def run_training(args):
    """Main function"""
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

    # Determine device
    if args.cpu is True:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: pre-analysis for CK Tests and ITS Plots
    # Get state transition probabilities, graph embeddings and attention scores
    probs, embeddings, attentions, edge_indices = analyze_vampnet_outputs(model=model,
                                                                          data_loader=loader,
                                                                          save_folder=paths['model_dir'],
                                                                          batch_size=args.batch_size if hasattr(args,
                                                                                                                'batch_size') else 32,
                                                                          device=device,
                                                                          return_tensors=False
                                                                          )

    # Get actual timestep of the trajectory
    # TODO: Check if this is correct!
    inferred_timestep = dataset._infer_timestep() / 1000  # Timestep in nanoseconds

    # Run Chapman Kolmogorow Test
    run_ck_analysis(
        probs=probs,
        save_dir=paths['plot_dir'],
        protein_name='ab42',#args.protein_name, #TODO: INCLUDE THIS AS AN ARGUMENT,
        lag_times_ns=args.lag_time,
        steps=10,
        stride=args.stride,
        timestep=inferred_timestep
    )
    print("CK test complete")

    # Calculate and plot implied timescales
    # TODO: Implement this
    max_tau = 250
    lags = [i for i in range(1, max_tau, 2)]
    its = get_its(probs, lags)
    # Using save_path instead of save_folder to match the function signature
    plot_its(its, lags, save_path=args.save_folder, ylog=False)
    np.save(os.path.join(args.save_folder, 'ITS.npy'), np.array(its))
    print("ITS calculation complete")






if __name__ == "__main__":
    # Parse arguments
    args = parse_train_args()

    run_training(args)
