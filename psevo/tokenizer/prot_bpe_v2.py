from copy import copy
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from collections import Counter
import hashlib
import pickle
import os
from pygv.dataset.vampnet_dataset_with_AA import VAMPNetDataset


class ProteinFrameInPiece:
    """Manages protein frame piece representation for BPE-like tokenization."""

    def __init__(self, graph_data, residue_features):
        """
        Initialize with PyTorch Geometric graph data and residue features.

        Args:
            graph_data: PyTorch Geometric Data object representing the protein frame
            residue_features: Dict mapping node indices to residue types (e.g., 'ALA', 'GLY', etc.)
        """
        self.graph = graph_data
        self.residue_features = residue_features
        self.num_nodes = graph_data.num_nodes

        # Initialize pieces: each residue starts as its own piece
        self.pieces, self.pieces_features = {}, {}
        for node_idx in range(self.num_nodes):
            residue_type = residue_features[node_idx]
            self.pieces[node_idx] = {node_idx: residue_type}
            self.pieces_features[node_idx] = residue_type

        # Create inverse mapping: node_id -> piece_id
        self.inversed_index = {}
        for node_idx in range(self.num_nodes):
            for key in self.pieces:
                piece = self.pieces[key]
                if node_idx in piece:
                    self.inversed_index[node_idx] = key

        self.dirty = True
        self.feature_to_pids = {}  # Cache for neighboring pieces

    def get_nei_pieces(self):
        """Find all possible neighboring piece combinations for merging."""
        nei_pieces, merge_pids = [], []
        edge_index = self.graph.edge_index

        for key in self.pieces:
            piece = self.pieces[key]
            local_nei_pid = []

            # Find neighboring pieces through graph connections
            for node_idx in piece:
                # Get neighbors of this node from edge_index
                neighbors = self._get_neighbors(node_idx, edge_index)

                for nei_idx in neighbors:
                    if nei_idx in piece or nei_idx > node_idx:
                        continue
                    local_nei_pid.append(self.inversed_index[nei_idx])

            # Create merged pieces for each neighbor
            local_nei_pid = set(local_nei_pid)
            for nei_pid in local_nei_pid:
                new_piece = copy(piece)
                new_piece.update(self.pieces[nei_pid])
                nei_pieces.append(new_piece)
                merge_pids.append((key, nei_pid))

        return nei_pieces, merge_pids

    def _get_neighbors(self, node_idx, edge_index):
        """Get neighbors of a node from edge_index tensor."""
        neighbors = []
        # Find edges where node_idx is the source
        source_mask = edge_index[0] == node_idx
        neighbors.extend(edge_index[1][source_mask].tolist())

        # Find edges where node_idx is the target
        target_mask = edge_index[1] == node_idx
        neighbors.extend(edge_index[0][target_mask].tolist())

        return list(set(neighbors))  # Remove duplicates

    def get_nei_features(self):
        """Get feature representations of all possible neighboring piece merges."""
        if self.dirty:
            nei_pieces, merge_pids = self.get_nei_pieces()
            nei_features, self.feature_to_pids = [], {}

            for i, piece in enumerate(nei_pieces):
                # Create a feature representation for this piece
                feature_str = self._piece_to_feature_string(piece)
                nei_features.append(feature_str)
                self.feature_to_pids.setdefault(feature_str, [])
                self.feature_to_pids[feature_str].append(merge_pids[i])

            self.dirty = False
        else:
            nei_features = list(self.feature_to_pids.keys())

        return nei_features

    def _piece_to_feature_string(self, piece):
        """Convert a piece (set of residues) to a feature string."""
        # Sort residues by their indices for consistency
        sorted_nodes = sorted(piece.keys())
        residue_types = [piece[node] for node in sorted_nodes]

        # Create a canonical string representation
        # Include both the residue composition and connectivity pattern
        feature_parts = []

        # Add residue composition (sorted for consistency)
        residue_counter = Counter(residue_types)
        for residue_type in sorted(residue_counter.keys()):
            feature_parts.append("{}:{}".format(residue_type, residue_counter[residue_type]))

        # Add connectivity information (simplified)
        edge_count = self._count_internal_edges(piece)
        feature_parts.append("edges:{}".format(edge_count))

        # Add size information
        feature_parts.append("size:{}".format(len(piece)))

        return "_".join(feature_parts)

    def _count_internal_edges(self, piece):
        """Count edges within a piece."""
        piece_nodes = set(piece.keys())
        edge_count = 0
        edge_index = self.graph.edge_index

        for i in range(edge_index.size(1)):
            source, target = edge_index[0, i].item(), edge_index[1, i].item()
            if source in piece_nodes and target in piece_nodes:
                edge_count += 1

        return edge_count // 2  # Each edge is counted twice

    def merge(self, feature_str):
        """Merge pieces that form the specified feature pattern."""
        if self.dirty:
            self.get_nei_features()

        if feature_str in self.feature_to_pids:
            merge_pids = self.feature_to_pids[feature_str]
            for pid1, pid2 in merge_pids:
                if pid1 in self.pieces and pid2 in self.pieces:
                    # Merge piece2 into piece1
                    self.pieces[pid1].update(self.pieces[pid2])
                    self.pieces_features[pid1] = feature_str

                    # Update inverse mapping
                    for node_idx in self.pieces[pid2]:
                        self.inversed_index[node_idx] = pid1

                    # Remove piece2
                    del self.pieces[pid2]
                    del self.pieces_features[pid2]

        self.dirty = True

    def get_feature_pieces(self):
        """Get final pieces as (feature_string, node_indices) tuples."""
        res = []
        for pid in self.pieces_features:
            feature_str = self.pieces_features[pid]
            group_dict = self.pieces[pid]
            node_indices = list(group_dict.keys())
            res.append((feature_str, node_indices))
        return res


def protein_bpe(frames_dataset, vocab_len, vocab_path, topology, cpus=1):
    """Protein piece extraction using BPE-like algorithm."""
    print("Processing {} protein frames...".format(len(frames_dataset)))

    # Convert frames to ProteinFrameInPiece objects
    frame_objects = []

    for i in tqdm(range(len(frames_dataset)), desc="Converting frames"):
        # Get graph data
        graph_data = frames_dataset[i]

        # Extract residue features from topology using atom indices from the original dataset
        residue_features = {}
        for node_idx in range(graph_data.num_nodes):
            # Get the actual atom index from the original VAMPNet dataset
            # Access atom_indices directly from the original dataset that created frames_dataset
            atom_idx = frames_dataset.atom_indices[node_idx]  # Use the original dataset variable

            # Get residue information from topology
            atom = frames_dataset.topology.atom(atom_idx)  # Use dataset.topology
            residue = atom.residue
            residue_features[node_idx] = residue.name

        frame_obj = ProteinFrameInPiece(graph_data, residue_features)
        frame_objects.append(frame_obj)

    # Initialize vocabulary with individual residues
    selected_features, details = [], {}

    # Collect all unique residue types
    all_residue_types = set()
    for frame_obj in frame_objects:
        for residue_type in frame_obj.residue_features.values():
            all_residue_types.add(residue_type)

    selected_features = list(all_residue_types)
    for residue_type in selected_features:
        details[residue_type] = [1, 0]  # [size, frequency]

    # Calculate residue frequencies
    for frame_obj in frame_objects:
        for residue_type in frame_obj.residue_features.values():
            if residue_type in details:
                details[residue_type][1] += 1

    # BPE process: iteratively find and merge most frequent patterns
    add_len = vocab_len - len(selected_features)

    for i in tqdm(range(add_len), desc="BPE iterations"):
        # Count frequencies across all frames
        freqs = {}
        updated_frames = []

        for frame_obj in frame_objects:
            nei_features = frame_obj.get_nei_features()
            for feature in nei_features:
                freqs.setdefault(feature, 0)
                freqs[feature] += 1
            updated_frames.append(frame_obj)

        frame_objects = updated_frames

        # Find most frequent pattern
        max_cnt, merge_feature = 0, ''
        for feature in freqs:
            cnt = freqs[feature]
            if cnt > max_cnt:
                max_cnt = cnt
                merge_feature = feature

        if max_cnt == 0:
            print("No more patterns to merge at iteration {}".format(i))
            break

        # Apply merge to all frames
        for frame_obj in frame_objects:
            frame_obj.merge(merge_feature)

        selected_features.append(merge_feature)
        # Estimate size based on feature complexity
        feature_size = len(merge_feature.split('_'))
        details[merge_feature] = [feature_size, max_cnt]

    print('Sorting vocab by feature complexity')
    selected_features.sort(key=lambda x: details[x][0], reverse=True)

    # Save vocabulary
    with open(vocab_path, 'w') as fout:
        for feature in selected_features:
            size, freq = details[feature]
            fout.write('{}\t{}\t{}\n'.format(feature, size, freq))

    return selected_features, details


# Usage example for your VAMPNet dataset:
# TODO: Remove
def create_protein_tokenizer_from_vampnet(vampnet_dataset, vocab_size=500, vocab_path="protein_vocab.txt"):
    """
    Create a protein tokenizer from VAMPNet dataset.

    Args:
        vampnet_dataset: Your VAMPNetDataset instance
        vocab_size: Size of vocabulary to create
        vocab_path: Path to save vocabulary
    """

    # Get frames dataset with amino acid encoding
    frames_dataset = vampnet_dataset.get_AA_frames(return_pairs=False)

    # Run BPE to create vocabulary
    vocab, details = protein_bpe(
        frames_dataset,
        vocab_size,
        vocab_path,
        vampnet_dataset.topology,  # Pass topology for residue extraction
        cpus=1
    )

    return vocab, details

