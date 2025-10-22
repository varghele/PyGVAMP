from copy import copy
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from collections import Counter
import hashlib
import pickle
import os


class ProteinFrameInPiece:
    """Manages protein frame piece representation for BPE-like tokenization."""

    def __init__(self, graph_data, amino_acid_features):
        """
        Initialize with PyTorch Geometric graph data and amino acid features.

        Args:
            graph_data: PyTorch Geometric Data object representing the protein frame
            amino_acid_features: Dict mapping node indices to amino acid types/features
        """
        self.graph = graph_data
        self.amino_acid_features = amino_acid_features
        self.num_nodes = graph_data.num_nodes

        # Initialize pieces: each amino acid starts as its own piece
        self.pieces, self.pieces_features = {}, {}
        for node_idx in range(self.num_nodes):
            aa_type = amino_acid_features[node_idx]
            self.pieces[node_idx] = {node_idx: aa_type}
            self.pieces_features[node_idx] = aa_type

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
        """Convert a piece (set of amino acids) to a feature string."""
        # Sort amino acids by their indices for consistency
        sorted_nodes = sorted(piece.keys())
        aa_types = [piece[node] for node in sorted_nodes]

        # Create a canonical string representation
        # Include both the amino acid types and their connectivity pattern
        feature_parts = []

        # Add amino acid composition
        aa_counter = Counter(aa_types)
        for aa_type in sorted(aa_counter.keys()):
            feature_parts.append(f"{aa_type}:{aa_counter[aa_type]}")

        # Add connectivity information (simplified)
        edge_count = self._count_internal_edges(piece)
        feature_parts.append(f"edges:{edge_count}")

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


def freq_cnt_protein(frame_obj):
    """Count frequency of neighboring feature patterns for parallel processing."""
    freqs = {}
    nei_features = frame_obj.get_nei_features()
    for feature in nei_features:
        freqs.setdefault(feature, 0)
        freqs[feature] += 1
    return freqs, frame_obj


def protein_bpe(frames_dataset, vocab_len, vocab_path, cpus=1):
    """Protein piece extraction using BPE-like algorithm."""
    print(f'Processing {len(frames_dataset)} protein frames...')

    # Convert frames to ProteinFrameInPiece objects
    frame_objects = []

    for i in tqdm(range(len(frames_dataset)), desc="Converting frames"):
        # Get graph data
        if hasattr(frames_dataset, 'get_graph'):
            graph_data = frames_dataset.get_graph(i)
        else:
            graph_data = frames_dataset[i]

        # Extract amino acid features from the dataset
        # This assumes you have amino acid information available
        amino_acid_features = {}
        for node_idx in range(graph_data.num_nodes):
            # You'll need to implement this based on your data structure
            # For now, using a placeholder - replace with actual amino acid extraction
            amino_acid_features[node_idx] = f"AA_{node_idx % 20}"  # Placeholder

        frame_obj = ProteinFrameInPiece(graph_data, amino_acid_features)
        frame_objects.append(frame_obj)

    # Initialize vocabulary with individual amino acids
    selected_features, details = [], {}

    # Collect all unique amino acid types
    all_aa_types = set()
    for frame_obj in frame_objects:
        for aa_type in frame_obj.amino_acid_features.values():
            all_aa_types.add(aa_type)

    selected_features = list(all_aa_types)
    for aa_type in selected_features:
        details[aa_type] = [1, 0]  # [size, frequency]

    # Calculate amino acid frequencies
    for frame_obj in frame_objects:
        for aa_type in frame_obj.amino_acid_features.values():
            if aa_type in details:
                details[aa_type][1] += 1

    # BPE process: iteratively find and merge most frequent patterns
    add_len = vocab_len - len(selected_features)

    # Note: Multiprocessing removed for simplicity - can be added back if needed
    for i in tqdm(range(add_len), desc="BPE iterations"):
        # Count frequencies across all frames
        freqs = {}
        updated_frames = []

        for frame_obj in frame_objects:
            freq, updated_frame = freq_cnt_protein(frame_obj)
            updated_frames.append(updated_frame)
            for key in freq:
                freqs.setdefault(key, 0)
                freqs[key] += freq[key]

        frame_objects = updated_frames

        # Find most frequent pattern
        max_cnt, merge_feature = 0, ''
        for feature in freqs:
            cnt = freqs[feature]
            if cnt > max_cnt:
                max_cnt = cnt
                merge_feature = feature

        if max_cnt == 0:
            print(f"No more patterns to merge at iteration {i}")
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
            fout.write(f'{feature}\t{size}\t{freq}\n')

    return selected_features, details


class ProteinTokenizer:
    """Protein piece tokenizer for BPE-based representation."""

    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as fin:
            lines = fin.read().strip().split('\n')

        # Build vocabulary
        self.vocab_dict = {}
        self.idx2piece, self.piece2idx = [], {}

        for line in lines:
            feature, size, freq = line.strip().split('\t')
            self.vocab_dict[feature] = (int(size), int(freq))
            self.piece2idx[feature] = len(self.idx2piece)
            self.idx2piece.append(feature)

        # Add special tokens
        self.pad, self.end = '<pad>', '<s>'
        for token in [self.pad, self.end]:
            self.piece2idx[token] = len(self.idx2piece)
            self.idx2piece.append(token)

    def tokenize(self, graph_data, amino_acid_features, return_idx=False):
        """Tokenize protein frame into pieces using greedy frequency-based merging."""

        frame_obj = ProteinFrameInPiece(graph_data, amino_acid_features)

        # Greedy merging based on vocabulary frequencies
        while True:
            nei_features = frame_obj.get_nei_features()
            max_freq, merge_feature = -1, ''

            for feature in nei_features:
                if feature not in self.vocab_dict:
                    continue
                freq = self.vocab_dict[feature][1]
                if freq > max_freq:
                    max_freq, merge_feature = freq, feature

            if max_freq == -1:
                break
            frame_obj.merge(merge_feature)

        res = frame_obj.get_feature_pieces()

        # Add start and end tokens
        res.insert(0, (self.end, []))
        res.append((self.end, []))

        if not return_idx:
            return res

        piece_idxs = [self.piece_to_idx(x[0]) for x in res]
        group_idxs = [x[1] for x in res]
        return piece_idxs, group_idxs

    def idx_to_piece(self, idx):
        return self.idx2piece[idx]

    def piece_to_idx(self, piece):
        return self.piece2idx[piece]

    def pad_idx(self):
        return self.piece2idx[self.pad]

    def end_idx(self):
        return self.piece2idx[self.end]

    def num_piece_type(self):
        return len(self.idx2piece)

    def __call__(self, graph_data, amino_acid_features, return_idx=False):
        return self.tokenize(graph_data, amino_acid_features, return_idx)

    def __len__(self):
        return len(self.idx2piece)


# Usage example for your VAMPNet dataset:
def create_protein_tokenizer_from_vampnet(vampnet_dataset, vocab_size=500, vocab_path="protein_vocab.txt"):
    """
    Create a protein tokenizer from VAMPNet dataset.

    Args:
        vampnet_dataset: Your VAMPNetDataset instance
        vocab_size: Size of vocabulary to create
        vocab_path: Path to save vocabulary
    """

    # Get frames dataset (individual frames, not pairs)
    frames_dataset = vampnet_dataset.get_AA_frames(return_pairs=False)

    # Run BPE to create vocabulary
    vocab, details = protein_bpe(frames_dataset, vocab_size, vocab_path, cpus=1)

    # Create tokenizer
    tokenizer = ProteinTokenizer(vocab_path)

    return tokenizer


# Example usage:
if __name__ == "__main__":
    # Assuming you have your VAMPNet dataset ready
    # vampnet_dataset = VAMPNetDataset(...)

    # Create tokenizer
    # tokenizer = create_protein_tokenizer_from_vampnet(
    #     vampnet_dataset,
    #     vocab_size=500,
    #     vocab_path="protein_vocab.txt"
    # )

    # Use tokenizer on a frame
    # graph_data = vampnet_dataset.get_graph(0)
    # amino_acid_features = {...}  # Extract from your data
    # tokens = tokenizer(graph_data, amino_acid_features)
    # print(f"Tokenized frame: {tokens}")

    pass
