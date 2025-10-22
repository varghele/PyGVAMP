from copy import copy
from rdkit.Chem import BondType

MAX_VALENCE = {'B': 3,
               'Br':1,
               'C':4,
               'Cl':1,
               'F':1,
               'I':1,
               'N':3,
               'O':2,
               'P':5,
               'S':6} #, 'Se':4, 'Si':4}

Bond_List = [None,
             BondType.SINGLE,
             BondType.DOUBLE,
             BondType.TRIPLE]  # aromatic bonds are not good


class GeneralVocab:
    """ General Chemical Vocabulary for atoms and bonds.
    This class provides bidirectional mappings between chemical elements/bonds
    and their corresponding indices for use in graph neural networks. It handles
    both standard chemical elements and special tokens for padding/masking.

    Key Features:
    - Atom vocabulary: Maps chemical elements (C, N, O, etc.) to indices
    - Bond vocabulary: Maps bond types (single, double, triple) to indices
    - Special token support: Handles padding and other special tokens
    - Valence information: Provides bond valence values for chemical validation

    Usage:
        vocab = GeneralVocab(atom_special=['<pad>'], bond_special=['<none>'])
        carbon_idx = vocab.atom_to_idx('C')
        single_bond_idx = vocab.bond_to_idx(BondType.SINGLE)
    """

    def __init__(self, atom_special=None, bond_special=None):
        """
        Initialize the chemical vocabulary.

        Args:
            atom_special (list, optional): Special atom tokens (e.g., padding tokens)
            bond_special (list, optional): Special bond tokens (e.g., no-bond tokens)
        """
        # =================================================================
        # ATOM VOCABULARY SETUP
        # =================================================================

        # Initialize with standard chemical elements from MAX_VALENCE dictionary
        # MAX_VALENCE contains mappings like {'C': 4, 'N': 3, 'O': 2, ...}
        self.idx2atom = list(MAX_VALENCE.keys())

        # Add special atom tokens if provided
        if atom_special is None:
            atom_special = []
        self.idx2atom += atom_special

        # Create bidirectional mapping: atom symbol ↔ index
        self.atom2idx = {atom: i for i, atom in enumerate(self.idx2atom)}

        # =================================================================
        # BOND VOCABULARY SETUP
        # =================================================================

        # Initialize with standard bond types from Bond_List
        # Bond_List contains RDKit bond types: [SINGLE, DOUBLE, TRIPLE, AROMATIC, ...]
        self.idx2bond = copy(Bond_List)

        # Add special bond tokens if provided
        if bond_special is None:
            bond_special = []
        self.idx2bond += bond_special

        # Create bidirectional mapping: bond type ↔ index
        self.bond2idx = {bond: i for i, bond in enumerate(self.idx2bond)}

        # =================================================================
        # STORE SPECIAL TOKENS FOR REFERENCE
        # =================================================================

        self.atom_special = atom_special
        self.bond_special = bond_special

    def idx_to_atom(self, idx):
        """
        Convert atom index to chemical element symbol.

        Args:
            idx (int): Atom index

        Returns:
            str: Chemical element symbol (e.g., 'C', 'N', 'O')
        """
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        """
        Convert chemical element symbol to index.

        Args:
            atom (str): Chemical element symbol (e.g., 'C', 'N', 'O')

        Returns:
            int: Atom index
        """
        return self.atom2idx[atom]

    def idx_to_bond(self, idx):
        """
        Convert bond index to RDKit bond type.

        Args:
            idx (int): Bond index

        Returns:
            BondType: RDKit bond type (e.g., BondType.SINGLE, BondType.DOUBLE)
        """
        return self.idx2bond[idx]

    def bond_to_idx(self, bond):
        """
        Convert RDKit bond type to index.

        Args:
            bond (BondType): RDKit bond type (e.g., BondType.SINGLE, BondType.DOUBLE)

        Returns:
            int: Bond index
        """
        return self.bond2idx[bond]

    def bond_idx_to_valence(self, idx):
        """
        Convert bond index to valence contribution.

        This method maps bond types to their valence contributions for
        chemical validation during molecular generation.

        Args:
            idx (int): Bond index

        Returns:
            int: Valence contribution (1 for single, 2 for double, 3 for triple, -1 for invalid)
        """
        bond_type = self.idx2bond[idx]

        # Map standard bond types to their valence contributions
        if bond_type == BondType.SINGLE:
            return 1
        elif bond_type == BondType.DOUBLE:
            return 2
        elif bond_type == BondType.TRIPLE:
            return 3
        else:
            # Invalid or special bond types (e.g., no-bond, aromatic)
            return -1

    def num_atom_type(self):
        """
        Get the total number of atom types in vocabulary.

        Returns:
            int: Number of atom types (including special tokens)
        """
        return len(self.idx2atom)

    def num_bond_type(self):
        """
        Get the total number of bond types in vocabulary.

        Returns:
            int: Number of bond types (including special tokens)
        """
        return len(self.idx2bond)

    def get_vocab_info(self):
        """
        Get comprehensive vocabulary information.

        Returns:
            dict: Vocabulary statistics and configuration
        """
        return {
            'num_atoms': self.num_atom_type(),
            'num_bonds': self.num_bond_type(),
            'atom_special_tokens': self.atom_special,
            'bond_special_tokens': self.bond_special,
            'standard_atoms': [atom for atom in self.idx2atom if atom not in self.atom_special],
            'standard_bonds': [bond for bond in self.idx2bond if bond not in self.bond_special]
        }

