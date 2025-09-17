import os
from tqdm import tqdm
from random import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from psevo.tokenizer.mol_bpe_old import Tokenizer
from psevo.utils.chem_utils import smiles2molecule
from psevo.utils.evaluation import get_normalized_property_scores


class BPEMolDataset(Dataset):
    """ Dataset for molecular BPE (Byte Pair Encoding) representation.
    This dataset processes molecular data through a three-stage pipeline:
    1. Raw representation: Convert molecules to basic graph format
    2. Adjacency matrix: Create dense adjacency representation
    3. Batch collation: Prepare data for neural network training

    The dataset handles molecular tokenization, graph construction,
    and property calculation for molecular generation tasks.
    """

    def __init__(self, fname, tokenizer):
        super(BPEMolDataset, self).__init__()
        self.root_path, self.file_path = os.path.split(fname)
        self.save_path = os.path.join(self.root_path, 'processed_data.pkl')
        self.tokenizer = tokenizer

        # Try to load preprocessed data, otherwise process from scratch
        try:
            self.data = torch.load(self.save_path)
        except FileNotFoundError:
            self.data = self.process()

    @staticmethod
    def process_step1(mol, tokenizer):
        """
        Convert molecule to raw graph representation.

        This step extracts basic molecular information:
        - Node features (atom types)
        - Edge connectivity and attributes
        - Molecular properties
        - Piece-based tokenization

        Args:
            mol (Chem.Mol): RDKit molecule object
            tokenizer: Molecular tokenizer for piece extraction

        Returns:
            dict: Raw molecular representation
        """
        # Extract atom features
        x = [tokenizer.chem_vocab.atom_to_idx(mol.GetAtomWithIdx(i).GetSymbol())
             for i in range(mol.GetNumAtoms())]

        # Extract bond information (store as single direction to save space)
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtom().GetIdx()
            end_idx = bond.GetEndAtom().GetIdx()
            bond_type = tokenizer.chem_vocab.bond_to_idx(bond.GetBondType())

            edge_index.append([begin_idx, end_idx])
            edge_attr.append(bond_type)

        # Calculate molecular properties
        properties = get_normalized_property_scores(mol)

        # Tokenize molecule into pieces
        pieces, groups = tokenizer(mol, return_idx=True)

        return {
            'x': x,                    # Atom types
            'edge_index': edge_index,  # Edge connectivity
            'edge_attr': edge_attr,    # Bond types
            'props': properties,       # Molecular properties
            'pieces': pieces,          # Piece sequence
            'groups': groups           # Atom groupings by piece
        }

    @staticmethod
    def process_step2(data, tokenizer):
        """
        Convert raw representation to adjacency matrix format.

        This step creates dense adjacency matrices and prepares
        data for edge prediction tasks by marking which edges
        should be predicted vs. which already exist within pieces.

        Args:
            data (dict): Raw molecular data from step 1
            tokenizer: Molecular tokenizer

        Returns:
            dict: Adjacency matrix representation
        """
        x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']

        # Create adjacency matrix (dense representation)
        none_bond_idx = tokenizer.chem_vocab.bond_to_idx(None)
        adjacency_matrix = [[none_bond_idx for _ in x] for _ in x]

        # Fill adjacency matrix with bond types (bidirectional)
        for i in range(len(edge_attr)):
            begin_idx, end_idx = edge_index[i]
            bond_type = edge_attr[i]
            adjacency_matrix[begin_idx][end_idx] = bond_type
            adjacency_matrix[end_idx][begin_idx] = bond_type

        # Initialize piece and position information for each atom
        x_pieces = [0 for _ in x]  # Piece ID for each atom
        x_pos = [0 for _ in x]     # Position in sequence for each atom
        pieces = data['pieces']

        # Create edge selection mask for inter-piece edge prediction
        edge_select = [[1 for _ in x] for _ in x]

        # Process each piece group
        for pos, group in enumerate(data['groups']):
            group_len = len(group)

            # Mark intra-piece edges as not to be predicted
            for i in range(group_len):
                for j in range(i, group_len):
                    atom_m, atom_n = group[i], group[j]
                    edge_select[atom_m][atom_n] = 0
                    edge_select[atom_n][atom_m] = 0

            # Assign piece and position information to atoms
            for atom_id in group:
                x_pieces[atom_id] = pieces[pos]
                x_pos[atom_id] = pos  # Position starts from 0

        return {
            'x': x,
            'ad_mat': adjacency_matrix,
            'props': data['props'],
            'pieces': pieces,
            'x_pieces': x_pieces,
            'x_pos': x_pos,
            'edge_select': edge_select
        }

    @staticmethod
    def process_step3(data_list, tokenizer, device='cpu'):
        """
        Collate batch data for neural network training.

        This step handles variable-length molecules by padding,
        creates batched tensors, and prepares edge prediction targets
        with proper class balancing.

        Args:
            data_list (list): List of processed molecular data
            tokenizer: Molecular tokenizer
            device (str): Device for tensor placement

        Returns:
            dict: Batched data ready for training
        """
        # Pad atom-level features
        atom_features, atom_lengths, atom_pieces, atom_positions = [], [], [], []

        for data in data_list:
            x = torch.tensor(data['x'], device=device)
            atom_features.append(x)
            atom_lengths.append(len(x))
            atom_pieces.append(torch.tensor(data['x_pieces'], device=device))
            atom_positions.append(torch.tensor(data['x_pos'], device=device))

        # Pad sequences to same length
        atom_features = pad_sequence(atom_features, batch_first=True,
                                   padding_value=tokenizer.atom_pad_idx())
        atom_pieces = pad_sequence(atom_pieces, batch_first=True,
                                 padding_value=tokenizer.pad_idx())
        atom_positions = pad_sequence(atom_positions, batch_first=True,
                                    padding_value=0)

        # Create atom mask for valid (non-padded) atoms
        batch_size, max_atoms = atom_features.shape
        atom_mask = torch.zeros(batch_size, max_atoms + 1, dtype=torch.long, device=device)
        atom_mask[torch.arange(batch_size, device=device), atom_lengths] = 1
        atom_mask = atom_mask.cumsum(dim=1)[:, :-1]  # Remove extra column

        # Process edges and create batched edge information
        edge_index, edge_attr, golden_edge, properties = [], [], [], []
        intra_piece_edge_indices = []
        edge_select = torch.zeros(batch_size, max_atoms, max_atoms, device=device)
        none_bond_idx = tokenizer.chem_vocab.bond_to_idx(None)

        for i, data in enumerate(data_list):
            adjacency_matrix = data['ad_mat']
            offset = max_atoms * i  # Offset for batching
            properties.append(data['props'])

            # Process adjacency matrix
            for m, row in enumerate(data['edge_select']):
                for n, should_predict in enumerate(row):
                    edge_select[i][m][n] = should_predict
                    bond_type = adjacency_matrix[m][n]
                    begin_idx, end_idx = m + offset, n + offset

                    # Add existing bonds to edge list
                    if bond_type != none_bond_idx:
                        edge_index.append([begin_idx, end_idx])
                        edge_attr.append(bond_type)

                        # Mark intra-piece edges for decoder
                        if should_predict == 0:
                            intra_piece_edge_indices.append(len(edge_index) - 1)

                    # Collect edges for prediction with class balancing
                    if should_predict == 1:
                        # Balance none-bond vs real bond classes
                        # Original ratio is heavily skewed (~0.022)
                        if bond_type != none_bond_idx or random() < 0.022:
                            golden_edge.append(bond_type)
                        else:
                            edge_select[i][m][n] = 0  # Remove from prediction

        # Pad piece sequences
        pieces = pad_sequence([torch.tensor(data['pieces'], dtype=torch.long, device=device)
                              for data in data_list],
                             batch_first=True, padding_value=tokenizer.pad_idx())

        # Convert edge data to tensors
        edge_attr = torch.tensor(edge_attr, dtype=torch.long, device=device)

        if len(edge_index):
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)

        return {
            'x': atom_features,                    # [batch_size, max_atoms]
            'x_pieces': atom_pieces,               # [batch_size, max_atoms]
            'x_pos': atom_positions,               # [batch_size, max_atoms]
            'atom_mask': atom_mask.bool(),         # [batch_size, max_atoms]
            'pieces': pieces,                      # [batch_size, seq_len]
            'edge_index': edge_index,              # [2, num_edges]
            'edge_attr': F.one_hot(edge_attr, num_classes=tokenizer.chem_vocab.num_bond_type()),
            'edge_select': edge_select.bool(),     # [batch_size, max_atoms, max_atoms]
            'golden_edge': torch.tensor(golden_edge, dtype=torch.long),
            'in_piece_edge_idx': intra_piece_edge_indices,
            'props': torch.tensor(properties)      # [batch_size, num_props]
        }

    def process(self):
        """Process raw SMILES data into molecular representations."""
        file_path = os.path.join(self.root_path, self.file_path)

        # Load SMILES strings
        with open(file_path, 'r') as fin:
            lines = fin.readlines()
        smiles_list = [s.strip('\n') for s in lines]

        # Convert SMILES to molecular data
        data_list = []
        for smi in tqdm(smiles_list):
            mol = smiles2molecule(smi, kekulize=True)
            if mol is None:
                continue
            data_list.append(self.process_step1(mol, self.tokenizer))

        # Save processed data
        torch.save(data_list, self.save_path)
        return data_list

    def __getitem__(self, idx):
        """Get a single data item and convert to adjacency format."""
        data = self.data[idx]
        return self.process_step2(data, self.tokenizer)

    def __len__(self):
        """Get dataset size."""
        return len(self.data)

    def collate_fn(self, data_list):
        """Custom collate function for batching variable-length molecules."""
        return self.process_step3(data_list, self.tokenizer)


def get_dataloader(fname, tokenizer, batch_size, shuffle=False, num_workers=4):
    """ Create a DataLoader for molecular BPE dataset.
    Args:
        fname (str): Path to SMILES file
        tokenizer: Molecular tokenizer
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes

    Returns:
        DataLoader: PyTorch DataLoader for molecular data
    """
    dataset = BPEMolDataset(fname, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=dataset.collate_fn, num_workers=num_workers)

if __name__ == '__main__':
    import sys
    # Example usage and testing
    if len(sys.argv) >= 3:
        tokenizer = Tokenizer(sys.argv[2])
        dataloader = get_dataloader(sys.argv[1], tokenizer, batch_size=1, shuffle=True)

        # Test data loading and examine batch structure
        for batch in tqdm(dataloader):
            print("Batch keys:", batch.keys())
            print("Atom features shape:", batch['x'].shape)
            print("Edge index shape:", batch['edge_index'].shape)
            print("Pieces shape:", batch['pieces'].shape)
            break


