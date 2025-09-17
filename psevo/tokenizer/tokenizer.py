import numpy as np
from psevo.utils.chem_utils import smiles2molecule
from psevo.utils.token_utils import GeneralVocab
from psevo.tokenizer.mol_bpe import MolInPiece


class Tokenizer:
    """ Molecular Piece-based Tokenizer using BPE-like approach.
    This tokenizer breaks molecules into meaningful molecular pieces/fragments
    using a greedy merging strategy based on pre-computed vocabulary frequencies.
    It implements a piece-based representation for molecular generation.

    Key Features:
    - Frequency-based piece merging (BPE-like approach)
    - Multi-level vocabulary (pieces + atoms)
    - Special tokens for padding and sequence boundaries
    - Random piece ordering for sequence generation
    """

    def __init__(self, vocab_path):
        """
        Initialize tokenizer with vocabulary from file.

        Vocabulary file format: SMILES_piece \t atom_count \t frequency

        Args:
            vocab_path (str): Path to vocabulary file
        """
        # Load vocabulary from file
        with open(vocab_path, 'r') as fin:
            lines = fin.read().strip().split('\n')

        # Initialize vocabulary structures
        self.vocab_dict = {}        # piece_smiles -> (atom_count, frequency)
        self.idx2piece = []         # index -> piece_smiles
        self.piece2idx = {}         # piece_smiles -> index

        # Parse vocabulary file
        for line in lines:
            smi, atom_num, freq = line.strip().split('\t')
            self.vocab_dict[smi] = (int(atom_num), int(freq))
            self.piece2idx[smi] = len(self.idx2piece)
            self.idx2piece.append(smi)

        # Add special tokens
        self.pad, self.end = '<pad>', '<s>'
        for smi in [self.pad, self.end]:
            self.piece2idx[smi] = len(self.idx2piece)
            self.idx2piece.append(smi)

        # Initialize atom-level vocabulary
        self.atom_pad = '<pad>'
        self.chem_vocab = GeneralVocab(atom_special=[self.atom_pad])

    def tokenize(self, mol, return_idx=False):
        """
        Tokenize molecule into pieces using greedy frequency-based merging.

        Algorithm:
        1. Start with individual atoms
        2. Find all possible neighboring piece merges
        3. Select merge that creates highest frequency piece in vocabulary
        4. Repeat until no more merges possible
        5. Add start/end tokens and randomize order

        Args:
            mol (str or Chem.Mol): Input molecule (SMILES string or RDKit mol)
            return_idx (bool): Whether to return indices instead of SMILES

        Returns:
            list: Piece sequence with start/end tokens
            tuple: (piece_indices, atom_groups) if return_idx=True
        """
        # Convert SMILES to molecule if needed
        if isinstance(mol, str):
            mol = smiles2molecule(mol)

        rdkit_mol = mol
        mol = MolInPiece(mol)  # Wrapper for piece-based operations

        # Greedy merging loop
        while True:
            # Get all possible neighboring pieces that can be merged
            neighboring_smiles = mol.get_nei_smis()
            max_freq, merge_smi = -1, ''

            # Find the most frequent piece in vocabulary
            for smi in neighboring_smiles:
                if smi not in self.vocab_dict:
                    continue
                freq = self.vocab_dict[smi][1]  # Get frequency
                if freq > max_freq:
                    max_freq, merge_smi = freq, smi

            # Stop if no valid merges found
            if max_freq == -1:
                break

            # Perform the merge
            mol.merge(merge_smi)

        # Get final pieces and their atom groups
        result = mol.get_smis_pieces()

        # Alternative ordering strategies (commented out in original)
        # Sort by atom count (largest first)
        # smi_atom_cnt = {smi: cnt_atom(smi) for smi, _ in result}
        # result.sort(key=lambda x: smi_atom_cnt[x[0]], reverse=True)

        # Extended Morgan BFS ordering (commented out in original)
        # This would create a more systematic ordering based on graph structure
        aid2pid = {}
        for pid, piece in enumerate(result):
            _, aids = piece
            for aid in aids:
                aid2pid[aid] = pid

        # Construct adjacency matrix between pieces
        adjacency_matrix = [[0 for _ in result] for _ in result]
        for aid in range(rdkit_mol.GetNumAtoms()):
            atom = rdkit_mol.GetAtomWithIdx(aid)
            for neighbor in atom.GetNeighbors():
                neighbor_id = neighbor.GetIdx()
                i, j = aid2pid[aid], aid2pid[neighbor_id]
                if i != j:  # Different pieces
                    adjacency_matrix[i][j] = adjacency_matrix[j][i] = 1

        # Could apply BFS-based ordering here (commented out)
        # order_list, _ = bfs_morgan_order_extended_by_admat(adjacency_matrix)
        # result = [result[i] for i in order_list]

        # Current approach: random shuffling
        np.random.shuffle(result)

        # Add start and end tokens
        result.insert(0, (self.end, []))  # Start token
        result.append((self.end, []))     # End token

        # Return based on requested format
        if not return_idx:
            return result

        # Convert to indices
        piece_indices = [self.piece_to_idx(x[0]) for x in result]
        group_indices = [x[1] for x in result]
        return piece_indices, group_indices

    def idx_to_piece(self, idx):
        """Convert piece index to SMILES string."""
        return self.idx2piece[idx]

    def piece_to_idx(self, piece):
        """Convert SMILES piece to index."""
        return self.piece2idx[piece]

    def pad_idx(self):
        """Get padding token index."""
        return self.piece2idx[self.pad]

    def end_idx(self):
        """Get end-of-sequence token index."""
        return self.piece2idx[self.end]

    def atom_pad_idx(self):
        """Get atom-level padding token index."""
        return self.chem_vocab.atom_to_idx(self.atom_pad)

    def num_piece_type(self):
        """Get total number of piece types in vocabulary."""
        return len(self.idx2piece)

    def num_atom_type(self):
        """Get total number of atom types in vocabulary."""
        return self.chem_vocab.num_atom_type()

    def __call__(self, mol, return_idx=False):
        """Make tokenizer callable."""
        return self.tokenize(mol, return_idx)

    def __len__(self):
        """Get vocabulary size."""
        return len(self.idx2piece)

