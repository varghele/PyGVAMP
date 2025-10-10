import os
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from collections import Counter
import mdtraj as md
from pygv.dataset.vampnet_dataset_with_AA import VAMPNetDataset
from psevo.tokenizer.prot_bpe_v2 import ProteinFrameInPiece, protein_bpe


def find_xtc_files(base_path):
    """Find all .xtc files in the given directory and subdirectories."""
    xtc_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.xtc'):
                xtc_files.append(os.path.join(root, file))
    return xtc_files


def extract_residue_features_from_topology(topology, atom_indices):
    """Extract residue names for each atom index."""
    residue_features = {}
    for node_idx, atom_idx in enumerate(atom_indices):
        residue = topology.atom(atom_idx).residue
        residue_features[node_idx] = residue.name
    return residue_features


def test_protein_tokenizer():
    """Test the protein tokenizer on real trajectory data using psevo classes."""

    # Specify the base directory path
    base_path = "/home/iwe81/PycharmProjects/DDVAMP/datasets/ab42/trajectories/red/"

    # Find all the .xtc files
    traj_files = find_xtc_files(base_path)
    print("Found {} trajectory files".format(len(traj_files)))

    # Topology file
    top = os.path.join(base_path, "topol.pdb")

    # Data processing settings
    selection = 'name CA'
    stride = 1
    lag_time = 20.0
    n_neighbors = 4
    node_embedding_dim = 16
    gaussian_expansion_dim = 16
    cache_dir = './area53/cache'
    use_cache = True

    # Create VAMPNet dataset
    print("Creating VAMPNet dataset...")
    dataset = VAMPNetDataset(
        trajectory_files=traj_files,
        topology_file=top,
        lag_time=lag_time,
        n_neighbors=n_neighbors,
        node_embedding_dim=node_embedding_dim,
        gaussian_expansion_dim=gaussian_expansion_dim,
        selection=selection,
        stride=stride,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    # Get frames dataset instead of time-lagged pairs dataset
    print("Getting frames dataset...")
    frames_dataset = dataset.get_frames_dataset(return_pairs=False)

    print("Frames dataset size: {}".format(len(frames_dataset)))
    print("Number of atoms per frame: {}".format(dataset.n_atoms))

    # Test with a small subset first
    vocab_size = 100
    vocab_path = "protein_vocab_test.txt"

    # Limit to first 100 frames for testing
    class SubsetDataset:
        def __init__(self, parent_dataset, vampnet_dataset, max_frames=100):
            self.parent = parent_dataset
            self.vampnet_dataset = vampnet_dataset
            self.max_frames = min(max_frames, len(parent_dataset))

        def __len__(self):
            return self.max_frames

        def __getitem__(self, idx):
            if idx >= self.max_frames:
                raise IndexError("Index out of range")
            return self.parent[idx]

    # Create subset for testing
    subset_frames = SubsetDataset(frames_dataset, dataset, max_frames=100)

    print("Testing with {} frames...".format(len(subset_frames)))

    # Convert frames to ProteinFrameInPiece objects for BPE
    print("Converting frames to ProteinFrameInPiece objects...")
    frame_objects = []

    for i in tqdm(range(len(subset_frames)), desc="Converting frames"):
        # Get graph data
        graph_data = subset_frames[i]

        # Extract residue features from topology
        residue_features = extract_residue_features_from_topology(
            dataset.topology,
            dataset.atom_indices
        )

        # Create ProteinFrameInPiece object
        frame_obj = ProteinFrameInPiece(graph_data, residue_features)
        frame_objects.append(frame_obj)

    print("Created {} ProteinFrameInPiece objects".format(len(frame_objects)))

    # Run protein BPE using the psevo implementation
    print("Running protein BPE...")
    vocab, details = protein_bpe(
        #frame_objects,  # Pass the frame objects directly # TODO: this ain't great
        frames_dataset = frames_dataset,
        vocab_len = vocab_size,
        vocab_path = vocab_path,
        topology=dataset.topology,
        cpus=1
    )

    print("\nVocabulary creation completed!")
    print("Vocabulary size: {}".format(len(vocab)))
    print("Vocabulary saved to: {}".format(vocab_path))

    # Display some vocabulary statistics
    print("\nVocabulary Statistics:")
    print("=" * 50)

    # Count different types of features
    single_residues = [f for f in vocab if ':' not in f or f.count(':') == 1]
    complex_features = [f for f in vocab if f.count(':') > 1]

    print("Single residues: {}".format(len(single_residues)))
    print("Complex features: {}".format(len(complex_features)))

    # Show first 10 vocabulary items
    print("\nFirst 10 vocabulary items:")
    for i, feature in enumerate(vocab[:10]):
        size, freq = details[feature]
        print("  {}: {} (size: {}, freq: {})".format(i + 1, feature, size, freq))

    # Show some complex features
    if complex_features:
        print("\nSome complex features:")
        for i, feature in enumerate(complex_features[:5]):
            size, freq = details[feature]
            print("  {}: {} (size: {}, freq: {})".format(i + 1, feature, size, freq))

    # Test tokenization on a single frame
    print("\nTesting tokenization on first frame...")
    try:
        # Get the first frame object
        test_frame = frame_objects[0]

        # Get neighboring features (this tests the tokenization process)
        nei_features = test_frame.get_nei_features()
        print("Found {} neighboring features for tokenization".format(len(nei_features)))

        if nei_features:
            print("Sample neighboring features:")
            for i, feature in enumerate(nei_features[:3]):
                print("  {}: {}".format(i + 1, feature))

        # Get final pieces
        final_pieces = test_frame.get_feature_pieces()
        print("Final pieces: {} fragments".format(len(final_pieces)))

        if final_pieces:
            print("Sample final pieces:")
            for i, (feature_str, node_indices) in enumerate(final_pieces[:3]):
                print("  {}: {} (nodes: {})".format(i + 1, feature_str, node_indices))

    except Exception as e:
        print("Error during tokenization test: {}".format(str(e)))

    print("\nTest completed successfully!")

    return vocab, details


if __name__ == "__main__":
    try:
        vocab, details = test_protein_tokenizer()
        print("\n✅ SUCCESS: Protein tokenizer test completed!")

    except Exception as e:
        print("❌ ERROR: {}".format(str(e)))
        import traceback

        traceback.print_exc()
