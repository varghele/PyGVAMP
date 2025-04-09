import os
import torch
import numpy as np
from typing import List
from torch_geometric.loader import DataLoader
from pygv.utils.analysis import analyze_vampnet_outputs
from pygv.utils.plotting import plot_state_populations, plot_state_evolution, plot_transition_probabilities
from pygv.utils.ck import run_ck_analysis
from pygv.dataset.vampnet_dataset import VAMPNetDataset
from pygv.encoder.schnet_wo_embed import SchNetEncoder
from pygv.vampnet.vampnet import VAMPNet
import glob

def run_complete_vampnet_analysis(
        model,
        data_loader: DataLoader,
        output_dir: str = "./analysis_results",
        protein_name: str = "protein",
        batch_size: int = 32,
        lag_times_for_transition: List[int] = [1, 5, 10],
        lag_times_for_ck: List[int] = [1, 5, 10],
        ck_steps: int = 5,
        timestep: float = 1.0,
        time_unit: str = "ns",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run a comprehensive analysis of a trained VAMPNet model.

    Parameters
    ----------
    model : VAMPNet
        Trained VAMPNet model
    data_loader : DataLoader or List[DataLoader]
        PyG DataLoader(s) containing trajectory data as graphs
    output_dir : str, optional
        Base directory for saving analysis results
    protein_name : str, optional
        Name of the protein for file naming
    batch_size : int, optional
        Batch size for processing
    lag_times_for_transition : List[int], optional
        Lag times for transition probability matrices
    lag_times_for_ck : List[int], optional
        Lag times for Chapman-Kolmogorov tests
    ck_steps : int, optional
        Number of steps for Chapman-Kolmogorov tests
    timestep : float, optional
        Time between frames in time_unit
    time_unit : str, optional
        Time unit for plots
    device : str, optional
        Device to run analysis on

    Returns
    -------
    dict
        Dictionary containing all analysis results
    """
    print(f"\n{'=' * 80}\nStarting comprehensive VAMPNet analysis\n{'=' * 80}\n")

    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for different analyses
    embeddings_dir = os.path.join(output_dir, "embeddings")
    states_dir = os.path.join(output_dir, "states")
    transitions_dir = os.path.join(output_dir, "transitions")
    ck_dir = os.path.join(output_dir, "chapman_kolmogorov")

    for directory in [embeddings_dir, states_dir, transitions_dir, ck_dir]:
        os.makedirs(directory, exist_ok=True)

    # Step 1: Extract model outputs (embeddings, probabilities, attention)
    print(f"\n{'=' * 40}\nExtracting model outputs\n{'=' * 40}")
    probs, embeddings, attentions = analyze_vampnet_outputs(
        model=model,
        data_loader=data_loader,
        save_folder=embeddings_dir,
        batch_size=batch_size,
        device=device
    )

    # Step 2: Plot state populations
    print(f"\n{'=' * 40}\nAnalyzing state populations\n{'=' * 40}")
    populations = plot_state_populations(
        probs=probs,
        save_dir=states_dir,
        protein_name=protein_name
    )

    # Step 3: Plot state evolution over time
    print(f"\n{'=' * 40}\nAnalyzing state evolution\n{'=' * 40}")
    plot_state_evolution(
        probs=probs,
        save_dir=states_dir,
        protein_name=protein_name,
        timestep=timestep,
        time_unit=time_unit
    )

    # Step 4: Plot state assignments and transitions
    """print(f"\n{'=' * 40}\nAnalyzing state assignments\n{'=' * 40}")
    plot_state_assignments(
        probs=probs,
        save_dir=states_dir,
        protein_name=protein_name,
        timestep=timestep,
        time_unit=time_unit
    )"""

    # Step 5: Calculate transition matrices for different lag times
    print(f"\n{'=' * 40}\nCalculating transition matrices\n{'=' * 40}")
    transition_matrices = {}
    for lag in lag_times_for_transition:
        print(f"Lag time: {lag}")
        trans_matrix, trans_matrix_no_self = plot_transition_probabilities(
            probs=probs,
            save_dir=transitions_dir,
            protein_name=protein_name,
            lag_time=lag,
            cmap_name='YlOrRd'
        )
        transition_matrices[lag] = {
            'full': trans_matrix,
            'no_self': trans_matrix_no_self
        }

    # Step 6: Run Chapman-Kolmogorov tests
    print(f"\n{'=' * 40}\nRunning Chapman-Kolmogorov tests\n{'=' * 40}")
    ck_results = run_ck_analysis(
        probs=probs,
        save_folder=output_dir,
        protein_name=protein_name,
        tau_values=lag_times_for_ck,
        steps=ck_steps,
        lag_time_unit=time_unit
    )

    # Step 7: Compile analysis summary
    print(f"\n{'=' * 40}\nCompiling analysis summary\n{'=' * 40}")

    # Extract state information
    n_states = probs[0].shape[1]

    # Collect metrics
    summary = {
        'model_name': type(model).__name__,
        'protein_name': protein_name,
        'n_states': n_states,
        'n_trajectories': len(probs),
        'trajectory_lengths': [p.shape[0] for p in probs],
        'state_populations': populations.tolist(),
        'dominant_state': int(np.argmax(populations)) + 1,
        'transition_matrices': transition_matrices,
        'chapman_kolmogorov': {
            lag: {
                'mse': np.mean((ck_results[lag]['predicted'] - ck_results[lag]['estimated']) ** 2),
                'mae': np.mean(np.abs(ck_results[lag]['predicted'] - ck_results[lag]['estimated']))
            } for lag in lag_times_for_ck
        }
    }

    # Save summary as JSON
    import json
    with open(os.path.join(output_dir, f"{protein_name}_analysis_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'=' * 40}\nAnalysis Summary\n{'=' * 40}")
    print(f"Protein: {protein_name}")
    print(f"Number of states: {n_states}")
    print(f"Number of trajectories: {len(probs)}")
    print(f"State populations:")
    for i, pop in enumerate(populations):
        print(f"  State {i + 1}: {pop:.3f}" + (" (dominant)" if i == np.argmax(populations) else ""))

    print("\nChapman-Kolmogorov test errors:")
    for lag in lag_times_for_ck:
        mse = summary['chapman_kolmogorov'][lag]['mse']
        mae = summary['chapman_kolmogorov'][lag]['mae']
        print(f"  Lag {lag} {time_unit}: MSE = {mse:.4f}, MAE = {mae:.4f}")

    print(f"\n{'=' * 80}\nAnalysis complete! Results saved to {output_dir}\n{'=' * 80}")

    return {
        'probs': probs,
        'embeddings': embeddings,
        'attentions': attentions,
        'populations': populations,
        'transition_matrices': transition_matrices,
        'ck_results': ck_results,
        'summary': summary
    }


def load_model_and_run_analysis(
        model_path: str,
        data_path: str,
        output_dir: str = "./analysis_results",
        protein_name: str = "ab42",
        batch_size: int = 32,
        lag_times_for_transition: list = [1, 5, 10],
        lag_times_for_ck: list = [1, 5, 10],
        ck_steps: int = 5,
        timestep: float = 1.0,
        time_unit: str = "ns",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")

    try:
        # Load the complete model
        model = VAMPNet.load_complete_model(model_path, map_location=device)

        print("Model loaded successfully!")
        model.eval()

        dataset_path = "/home/iwe81/PycharmProjects/DDVAMP/datasets/ab42/trajectories/trajectories/red/"
        # Find all .xtc files in the dataset directory
        xtc_files = glob.glob(os.path.join(dataset_path, "**/*.xtc"), recursive=True)
        topology_file = "/home/iwe81/PycharmProjects/DDVAMP/datasets/ab42/trajectories/topol.pdb"

        dataset = VAMPNetDataset(
        trajectory_files=xtc_files,
        topology_file=topology_file,
        lag_time=20,  # Lag time in ns
        n_neighbors=20,  # Number of neighbors for graph construction
        node_embedding_dim=16,
        gaussian_expansion_dim=8,
        selection="name CA",  # Select only C-alpha atoms
        stride=40,  # Take every 2nd frame to reduce dataset size
        cache_dir="testdata",
        use_cache=True
    )

        # Load the test data
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Run complete analysis
        results = run_complete_vampnet_analysis(
            model=model,
            data_loader=test_loader,
            output_dir=output_dir,
            protein_name=protein_name,
            batch_size=batch_size,
            lag_times_for_transition=lag_times_for_transition,
            lag_times_for_ck=lag_times_for_ck,
            ck_steps=ck_steps,
            timestep=timestep,
            time_unit=time_unit,
            device=device
        )

        print(f"Analysis complete! Results stored in: {output_dir}")
        return results

    except Exception as e:
        print(f"Error during model loading or analysis: {str(e)}")
        raise

load_model_and_run_analysis(model_path="mdl_save/mdl_full.pt",
                            data_path="None")

