#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Verification Script for VAMPNet
"""

import os
import sys
import argparse
import torch

# Add parent directory to sys.path to import from your package
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from training pipeline
from pygv.pipe.training import create_dataset_and_loader, create_model, train_model, setup_output_directory, save_config
from pygv.scores.vamp_score_v0 import VAMPScore


def create_test_args():
    """Create a simple argument namespace for testing"""
    args = argparse.Namespace()

    # Basic settings
    args.encoder_type = 'schnet'
    # args.encoder_type = 'meta'

    # Data settings
    args.traj_dir = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/')
    args.top = os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/traj_revgraphvamp_org/trajectories/red/topol.pdb')
    args.selection = 'name CA'
    args.stride = 1
    args.lag_time = 10.0
    args.n_neighbors = 10
    args.node_embedding_dim = 16
    args.gaussian_expansion_dim = 16

    # SchNet encoder settings
    args.node_dim = 16
    args.edge_dim = 16
    args.hidden_dim = 32
    args.output_dim = 16
    args.n_interactions = 4
    args.activation = 'tanh'
    args.use_attention = True

    # Classifier settings
    args.n_states = 4
    args.clf_hidden_dim = 32
    args.clf_num_layers = 2
    args.clf_dropout = 0.0
    args.clf_activation = 'relu'
    args.clf_norm = None

    # Embedding settings
    args.use_embedding=True
    args.embedding_in_dim=16
    args.embedding_hidden_dim=32
    args.embedding_out_dim=16
    args.embedding_num_layers=3
    args.embedding_dropout=0.0
    args.embedding_act='elu'
    args.embedding_norm=None


    # Training settings
    args.epochs = 5
    args.batch_size = 50000
    args.lr = 0.0005
    args.weight_decay = 1e-5
    args.clip_grad = None
    args.cpu = False  # Use CPU for testing

    # Output settings
    args.output_dir = './area53'
    args.cache_dir = './area53/cache'
    args.use_cache = False
    args.save_every = 0  # Don't save intermediates
    args.run_name = 'test_run'

    return args


def verify_model_training(model, loader, device='cpu', n_iterations=20):
    """
    Verify if a model can train by checking gradients, loss reduction, and parameter updates.

    Args:
        model: The PyTorch model to verify
        loader: DataLoader with training data
        device: Device to run the model on ('cpu' or 'cuda')
        n_iterations: Number of training iterations to run

    Returns:
        dict: Results of the verification including success status
    """
    print("\n" + "=" * 50)
    print("MODEL TRAINING VERIFICATION")
    print("=" * 50)

    # Set model to training mode
    model.train()

    # Get a mini-batch for testing
    try:
        batch = next(iter(loader))
        print(f"Successfully loaded a batch from the dataloader")
    except Exception as e:
        print(f"❌ ERROR: Failed to load data from dataloader: {str(e)}")
        return {'success': False, 'reason': 'dataloader_error'}

    # Create VAMPScore instance for loss calculation
    vamp_scorer = VAMPScore(method='VAMP2')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Save initial parameters
    param_copies = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_copies[name] = param.detach().clone()

    print(f"\n1. Checking forward pass...")
    try:
        # Forward pass - For VAMP, model outputs the encoding
        with torch.no_grad():
            model.to(device)
            data_t0, data_t1 = batch
            data_t0, data_t1 = data_t0.to(device), data_t1.to(device)
            encoding_t0, _ = model.encoder(data_t0.x, data_t0.edge_index, data_t0.edge_attr, data_t0.batch)
            encoding_t1, _ = model.encoder(data_t1.x, data_t1.edge_index, data_t1.edge_attr, data_t1.batch)

            # Apply classifier to get final embeddings
            chi_t0 = model.classifier_module(encoding_t0)
            chi_t1 = model.classifier_module(encoding_t1)



            # Calculate VAMP score (higher is better)
            vamp_score = vamp_scorer(chi_t0, chi_t1)
            vamp_loss = -vamp_score  # Negate for minimization

            print(f"  ✅ Initial VAMP score: {vamp_score.item():.6f}")
            print(f"  ✅ Initial VAMP loss: {vamp_loss.item():.6f}")

    except Exception as e:
        print(f"  ❌ Forward pass failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'reason': 'forward_pass_error'}

    print(f"\n2. Checking gradients and backpropagation...")
    try:
        # Forward and backward pass
        model.zero_grad()

        model.to(device)
        data_t0, data_t1 = batch
        data_t0, data_t1 = data_t0.to(device), data_t1.to(device)

        data_t0.x = model.embedding_module(data_t0.x)
        data_t1.x = model.embedding_module(data_t1.x)

        encoding_t0, _ = model.encoder(data_t0.x, data_t0.edge_index, data_t0.edge_attr, data_t0.batch)
        encoding_t1, _ = model.encoder(data_t1.x, data_t1.edge_index, data_t1.edge_attr, data_t1.batch)

        chi_t0 = model.classifier_module(encoding_t0)
        chi_t1 = model.classifier_module(encoding_t1)

        vamp_score = vamp_scorer(chi_t0, chi_t1)
        vamp_loss = -vamp_score

        vamp_loss.backward()

        # Check gradients
        has_grad = False
        has_nan_inf = False

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    print(f"  ⚠️ No gradient for {name}")
                    continue

                grad_norm = param.grad.norm().item()
                has_grad = has_grad or (grad_norm > 0)

                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"  ❌ NaN/Inf gradient in {name}")
                    has_nan_inf = True

        if has_nan_inf:
            print(f"  ❌ Found NaN/Inf gradients")
            return {'success': False, 'reason': 'nan_inf_gradients'}

        if not has_grad:
            print(f"  ❌ No gradients are flowing through the model")
            return {'success': False, 'reason': 'no_gradients'}

        print(f"  ✅ Gradients are flowing correctly")

    except Exception as e:
        print(f"  ❌ Backpropagation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'reason': 'backprop_error'}

    """print(f"\n3. Testing training loop for {n_iterations} iterations...")

    # Run a mini training loop
    losses = []
    scores = []

    try:
        for i in range(n_iterations):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            model.to(device)
            data_t0, data_t1 = batch
            data_t0, data_t1 = data_t0.to(device), data_t1.to(device)
            encoding_t0, _ = model.encoder(data_t0.x, data_t0.edge_index, data_t0.edge_attr, data_t0.batch)
            encoding_t1, _ = model.encoder(data_t1.x, data_t1.edge_index, data_t1.edge_attr, data_t1.batch)

            chi_t0 = model.classifier_module(encoding_t0)
            chi_t1 = model.classifier_module(encoding_t1)

            vamp_score = vamp_scorer(chi_t0, chi_t1)
            vamp_loss = -vamp_score

            # Backward pass
            vamp_loss.backward()

            # Update weights
            optimizer.step()

            # Store metrics
            losses.append(vamp_loss.item())
            scores.append(vamp_score.item())

            if i % 5 == 0 or i == n_iterations - 1:
                print(
                    f"  Iteration {i + 1}/{n_iterations}: VAMP Score = {vamp_score.item():.6f}, Loss = {vamp_loss.item():.6f}")

        # Plot the training curve
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('VAMP Loss vs. Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (-VAMP score)')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(scores)
        plt.title('VAMP Score vs. Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('VAMP Score')
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        os.makedirs('./area53/diagnostics', exist_ok=True)
        plt.savefig('./area53/diagnostics/training_verification.png')
        print(f"  ✅ Training curves saved to './area53/diagnostics/training_verification.png'")

    except Exception as e:
        print(f"  ❌ Training loop failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'reason': 'training_loop_error'}

    # Check if loss decreased and score increased
    is_loss_decreasing = losses[0] > losses[-1]
    is_score_increasing = scores[0] < scores[-1]"""

    print(f"\n3.5 Testing training loop for {n_iterations} iterations...")

    # Get consistent batch from loader
    all_data = []
    for batch in loader:
        all_data.append(batch)

    try:
        for i in range(n_iterations):
            for batch in loader:
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                model.to(device)
                data_t0, data_t1 = batch
                data_t0, data_t1 = data_t0.to(device), data_t1.to(device)

                data_t0.x = model.embedding_module(data_t0.x)
                data_t1.x = model.embedding_module(data_t1.x)

                encoding_t0, _ = model.encoder(data_t0.x, data_t0.edge_index, data_t0.edge_attr, data_t0.batch)
                encoding_t1, _ = model.encoder(data_t1.x, data_t1.edge_index, data_t1.edge_attr, data_t1.batch)

                chi_t0 = model.classifier_module(encoding_t0)
                chi_t1 = model.classifier_module(encoding_t1)

                vamp_score = vamp_scorer(chi_t0, chi_t1)
                vamp_loss = -vamp_score

                # Backward pass
                vamp_loss.backward()

                # Update weights
                optimizer.step()

                if i % 5 == 0 or i == n_iterations - 1:
                    print(
                        f"  Iteration {i + 1}/{n_iterations}: VAMP Score = {vamp_score.item():.6f}, Loss = {vamp_loss.item():.6f}")
    except Exception as e:
        print(f"  ❌ Training loop failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'reason': 'training_loop_error'}

    print(f"\n4. Analyzing training results...")
    print(f"  Initial VAMP score: {scores[0]:.6f}")
    print(f"  Final VAMP score: {scores[-1]:.6f}")
    print(f"  Score change: {scores[-1] - scores[0]:.6f}")
    print(f"  Score increased: {is_score_increasing}")

    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Loss change: {losses[0] - losses[-1]:.6f}")
    print(f"  Loss decreased: {is_loss_decreasing}")

    # Check parameter updates
    print(f"\n5. Checking parameter updates...")
    params_changed = 0
    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            # Compute relative change in parameter
            original = param_copies[name]
            change = torch.norm(param.data - original) / (torch.norm(original) + 1e-8)

            if change > 1e-5:
                params_changed += 1

    update_ratio = params_changed / max(1, total_params)
    print(f"  Parameters updated: {params_changed}/{total_params} ({update_ratio * 100:.1f}%)")

    # Final verdict
    can_train = is_score_increasing and update_ratio > 0.5

    print("\n" + "=" * 50)
    print(f"VERDICT: Model {'CAN' if can_train else 'CANNOT'} train properly")
    print("=" * 50)

    # Suggest fixes if needed
    if not can_train:
        print("\nPossible issues to check:")
        if not is_score_increasing:
            print("- VAMP score isn't increasing. Key issues could be:")
            print("  * Insufficient data variability between time-lagged samples")
            print("  * Too short lag time to capture meaningful transitions")
            print("  * Model capacity too low (try increasing hidden_dim or n_interactions)")
            print("  * Try different activation functions")

        if update_ratio <= 0.5:
            print("- Many parameters aren't updating. Potential causes:")
            print("  * Some layers aren't connected to the loss")
            print("  * Learning rate too low")
            print("  * Gradient vanishing in deeper layers")

    return {
        'success': can_train,
        'scores': scores,
        'losses': losses,
        'is_score_increasing': is_score_increasing,
        'is_loss_decreasing': is_loss_decreasing,
        'param_update_ratio': update_ratio
    }


def run_test():
    """Run a VAMPNet training test with verification"""
    # Create test arguments
    args = create_test_args()

    # Setup output directory
    paths = setup_output_directory(args)

    # Save configuration
    save_config(args, paths)

    # Create dataset and loader
    dataset, loader = create_dataset_and_loader(args)

    # Create model
    model = create_model(args, dataset)
    print(f"Created VAMPNet model with {sum(p.numel() for p in model.parameters())} parameters")

    # Verify the model can train
    verification_result = verify_model_training(model, loader, n_iterations=5000, device='cuda:0')

    if verification_result['success']:
        # If verification passed, proceed with full training
        print("\nModel verification successful! Proceeding with full training...")
        scores = train_model(args, model, loader, paths)
        print(f"Training completed successfully. Results saved to {paths['run_dir']}")
    else:
        print("\nModel verification failed. Please address the issues before running full training.")
        sys.exit(1)


if __name__ == "__main__":
    run_test()
