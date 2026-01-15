"""
Generate mock data and create a test visualization.

This script demonstrates how to use the MDTrajectoryVisualizer with synthetic data
that mimics realistic molecular dynamics analysis results.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path so we can import md_visualizer
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from md_visualizer import MDTrajectoryVisualizer


def generate_mock_embeddings(n_frames, n_states, seed=42):
    """
    Generate mock 2D embeddings with state-based clustering.

    Parameters
    ----------
    n_frames : int
        Number of frames to generate
    n_states : int
        Number of states/clusters
    seed : int
        Random seed for reproducibility

    Returns
    -------
    embeddings : np.ndarray
        2D embeddings of shape (n_frames, 2)
    state_assignments : np.ndarray
        State assignments of shape (n_frames,)
    """
    np.random.seed(seed)

    # Generate cluster centers
    angles = np.linspace(0, 2 * np.pi, n_states, endpoint=False)
    radius = 3.0
    centers = np.array([
        [radius * np.cos(angle), radius * np.sin(angle)]
        for angle in angles
    ])

    # Assign frames to states
    state_assignments = np.random.randint(0, n_states, n_frames)

    # Generate embeddings around cluster centers
    embeddings = []
    for state in state_assignments:
        center = centers[state]
        # Add noise
        noise = np.random.randn(2) * 0.5
        point = center + noise
        embeddings.append(point)

    return np.array(embeddings), state_assignments


def generate_mock_transition_matrix(n_states, seed=42):
    """
    Generate a realistic transition matrix.

    Parameters
    ----------
    n_states : int
        Number of states
    seed : int
        Random seed for reproducibility

    Returns
    -------
    transition_matrix : np.ndarray
        Transition matrix of shape (n_states, n_states)
    """
    np.random.seed(seed)

    # Create a matrix with higher probability for self-transitions
    # and neighboring states
    trans_matrix = np.zeros((n_states, n_states))

    for i in range(n_states):
        # High self-transition probability
        trans_matrix[i, i] = 0.6 + np.random.rand() * 0.2

        # Transitions to neighbors
        for j in range(n_states):
            if i != j:
                # Distance between states (circular)
                dist = min(abs(i - j), n_states - abs(i - j))
                # Closer states have higher transition probability
                trans_matrix[i, j] = np.exp(-dist) * 0.3 * np.random.rand()

        # Normalize rows to sum to 1
        trans_matrix[i, :] /= np.sum(trans_matrix[i, :])

    return trans_matrix


def generate_mock_attention(n_frames, n_residues, state_assignments, seed=42):
    """
    Generate mock attention values.

    Parameters
    ----------
    n_frames : int
        Number of frames
    n_residues : int
        Number of residues
    state_assignments : np.ndarray
        State assignments for each frame
    seed : int
        Random seed for reproducibility

    Returns
    -------
    attention : np.ndarray
        Attention values of shape (n_frames, n_residues)
    """
    np.random.seed(seed)

    n_states = len(np.unique(state_assignments))

    # Create state-specific attention patterns
    state_patterns = []
    for i in range(n_states):
        # Each state has different important residues
        pattern = np.random.rand(n_residues) * 0.3  # Base level

        # Add some high-attention regions
        n_important = np.random.randint(3, 8)
        important_indices = np.random.choice(n_residues, n_important, replace=False)
        pattern[important_indices] += np.random.rand(n_important) * 0.7

        state_patterns.append(pattern)

    # Generate attention for each frame based on its state
    attention = []
    for state in state_assignments:
        # Use state pattern with some noise
        frame_attention = state_patterns[state] + np.random.randn(n_residues) * 0.05
        frame_attention = np.clip(frame_attention, 0, 1)
        attention.append(frame_attention)

    return np.array(attention)


def generate_mock_pdb(n_residues=100):
    """
    Generate a simple mock PDB string for a linear peptide.

    Parameters
    ----------
    n_residues : int
        Number of residues

    Returns
    -------
    pdb_string : str
        PDB format string
    """
    pdb_lines = ["HEADER    MOCK PROTEIN STRUCTURE"]
    pdb_lines.append("TITLE     GENERATED FOR MD VISUALIZATION TESTING")

    atom_num = 1
    for res_num in range(1, n_residues + 1):
        # Simple CA-only model in a helix-like structure
        x = res_num * 3.8 * np.cos(res_num * 0.3)
        y = res_num * 3.8 * np.sin(res_num * 0.3)
        z = res_num * 1.5

        pdb_lines.append(
            f"ATOM  {atom_num:5d}  CA  ALA A{res_num:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
        atom_num += 1

    pdb_lines.append("END")

    return "\n".join(pdb_lines)


def main():
    """Main function to generate mock data and create visualization."""
    print("=" * 60)
    print("MD Trajectory Visualizer - Mock Data Generator")
    print("=" * 60)

    # Configuration
    n_residues = 76  # Ubiquitin has 76 residues
    lagtimes = [1, 5, 10]
    n_states = 5
    base_frames = 50

    # Create visualizer
    viz = MDTrajectoryVisualizer()

    print(f"\nGenerating mock data for {len(lagtimes)} timescales...")

    # Generate data for each lagtime
    for i, lagtime in enumerate(lagtimes):
        print(f"  Lagtime {lagtime}...", end=" ")

        # Number of frames decreases with lagtime (subsampling effect)
        n_frames = base_frames // (1 + i)

        # Generate data
        embeddings, state_assignments = generate_mock_embeddings(
            n_frames, n_states, seed=42 + i
        )

        frame_indices = np.sort(np.random.choice(
            base_frames * 10, n_frames, replace=False
        ))

        transition_matrix = generate_mock_transition_matrix(
            n_states, seed=42 + i
        )

        attention_values = generate_mock_attention(
            n_frames, n_residues, state_assignments, seed=42 + i
        )

        # Add to visualizer
        viz.add_timescale(
            lagtime=lagtime,
            embeddings=embeddings,
            frame_indices=frame_indices,
            state_assignments=state_assignments,
            transition_matrix=transition_matrix,
            attention_values=attention_values,
            metadata={'description': f'Mock data with lagtime {lagtime}'}
        )

        print(f"{n_frames} frames, {n_states} states")

    # Fetch real protein structure from RCSB
    print("\nFetching protein structure from RCSB (1UBQ - Ubiquitin)...")
    try:
        viz.set_protein_structure(pdb_path="1UBQ")
        print("Successfully loaded ubiquitin structure (76 residues)")
    except Exception as e:
        print(f"Warning: Could not fetch 1UBQ, using mock structure: {e}")
        pdb_string = generate_mock_pdb(n_residues)
        viz.set_protein_structure(pdb_string=pdb_string)

    # Print summary
    print("\n" + viz.get_summary())

    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)

    # Generate visualization
    output_path = output_dir / 'mock_visualization.html'
    print(f"\nGenerating visualization...")
    viz.generate(
        output_path=str(output_path),
        title="MD Trajectory Analysis - Mock Data"
    )

    print(f"\nVisualization saved to: {output_path}")
    print("\nOptions:")
    print("  1. Open the HTML file in a browser")
    print("  2. Or run with --show to launch automatically")

    # Check if --show flag is provided
    if '--show' in sys.argv:
        print("\nLaunching visualization in browser...")
        viz.show()


if __name__ == "__main__":
    main()
