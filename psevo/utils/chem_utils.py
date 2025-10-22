from rdkit import Chem
from rdkit.Chem.rdchem import BondType
# Initialize BFS data structures
from queue import Queue


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


def smiles2molecule(smiles: str, kekulize=True):
    """
    Convert a SMILES string to an RDKit molecule object with standardization.

    This function performs several important molecular preprocessing steps:
    1. Parses SMILES string into RDKit molecule object
    2. Removes stereochemical information for consistent representation
    3. Optionally converts to Kekule form (explicit double bonds)

    Mathematical/Chemical Background:
    - SMILES (Simplified Molecular Input Line Entry System) is a string notation
      for describing molecular structure using ASCII characters
    - Stereochemistry removal ensures consistent representation regardless of
      chirality specifications in the input SMILES
    - Kekulization converts aromatic systems to explicit single/double bond
      representation, which is often required for graph neural networks

    Args:
        smiles (str): SMILES string representation of the molecule
        kekulize (bool): Whether to convert aromatic bonds to explicit Kekule form
                        Default: True (recommended for GNN applications)

    Returns:
        Chem.Mol or None: RDKit molecule object if parsing successful, None otherwise

    Note:
        - Returns None if SMILES string is invalid or cannot be parsed
        - Kekulization may fail for some aromatic systems with unusual bonding
        - Stereochemistry removal affects chiral centers and double bond geometry
    """

    # Parse SMILES string to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)

    # Handle parsing failures
    if mol is None:
        return None

    # Remove stereochemical information
    Chem.RemoveStereochemistry(mol)

    #Optional Kekulization
    # Convert aromatic representation to explicit single/double bonds
    if kekulize:
        try:
            # Kekulize converts aromatic bonds (e.g., benzene ring) to alternating
            # single and double bonds, which is often required for:
            # - Graph neural networks that need explicit bond types
            # - Molecular property calculations
            # - Consistent bond order representation
            Chem.Kekulize(mol)
        except Chem.KekulizeException:
            # Some molecules cannot be kekulized (e.g., certain aromatic systems)
            # In such cases, we continue with the aromatic representation
            # This preserves the molecule rather than failing completely
            pass

    return mol


def molecule2smiles(mol):
    """ Convert an RDKit molecule object to SMILES string representation.
    This is a simple wrapper around RDKit's MolToSmiles function that
    provides a consistent interface for molecular string conversion.

    Args:
        mol (Chem.Mol): RDKit molecule object

    Returns:
        str: SMILES string representation of the molecule

    Note:
        This function uses RDKit's default SMILES generation settings.
        For more control over the output format, use Chem.MolToSmiles
        directly with specific parameters.
    """
    return Chem.MolToSmiles(mol)


def get_submol(mol, idx2atom):
    """ Extract a submolecule from a larger molecule based on specified atom indices.
    This function creates a new molecule containing only the atoms specified
    in the idx2atom dictionary, along with all bonds between those atoms.
    It's commonly used in molecular piece extraction and fragment analysis.

    Args:
        mol (Chem.Mol): Source RDKit molecule object
        idx2atom (dict): Dictionary mapping atom indices to atom symbols
                        {atom_index: atom_symbol, ...}

    Returns:
        Chem.Mol: New molecule containing only the specified atoms and their bonds
    """
    # Create editable molecule for building submolecule
    sub_mol = Chem.RWMol()

    # Mapping from original atom indices to new atom indices
    old_to_new_id = {}

    # Add atoms to submolecule
    for new_id, old_id in enumerate(idx2atom.keys()):
        # Get original atom and create new atom with same properties
        original_atom = mol.GetAtomWithIdx(old_id)
        new_atom = Chem.Atom(original_atom.GetSymbol())

        # Add atom to submolecule and store index mapping
        sub_mol.AddAtom(new_atom)
        old_to_new_id[old_id] = new_id

    # Add bonds between atoms in the submolecule
    for atom_id in idx2atom:
        atom = mol.GetAtomWithIdx(atom_id)

        # Check all bonds of this atom
        for bond in atom.GetBonds():
            # Get the other atom in this bond
            neighbor_id = bond.GetBeginAtomIdx()
            if neighbor_id == atom_id:
                neighbor_id = bond.GetEndAtomIdx()

            # Only add bond if:
            # 1. Neighbor is also in our submolecule
            # 2. We haven't already added this bond (neighbor_id < atom_id prevents duplicates)
            if neighbor_id in idx2atom and neighbor_id < atom_id:
                # Add bond using new atom indices
                sub_mol.AddBond(
                    old_to_new_id[atom_id],
                    old_to_new_id[neighbor_id],
                    bond.GetBondType()
                )

    # Convert to final molecule object
    sub_mol = sub_mol.GetMol()
    return sub_mol


def valence_check(aid1, aid2, edges1, edges2, new_edge, vocab, c1=0, c2=0):
    """
    Check if adding a new bond between two atoms would violate valence rules.

    This function validates whether a proposed bond formation is chemically valid
    by ensuring that neither atom would exceed its maximum allowed valence after
    the bond is added. This is crucial for generating chemically realistic molecules.

    Chemical Background:
    - Valence: The number of bonds an atom can form (combining capacity)
    - Each element has a maximum valence based on its electron configuration
    - Formal charges affect the effective valence capacity of atoms
    - Special handling for sulfur which can have valence 2 or 6

    Args:
        aid1 (int): Atom type index for first atom
        aid2 (int): Atom type index for second atom
        edges1 (list): List of bond type indices currently connected to atom 1
        edges2 (list): List of bond type indices currently connected to atom 2
        new_edge (int): Bond type index of the proposed new bond
        vocab: Chemical vocabulary object with conversion methods
        c1 (int): Formal charge of atom 1 (default: 0)
        c2 (int): Formal charge of atom 2 (default: 0)

    Returns:
        bool: True if the bond addition is chemically valid, False otherwise
    """

    # Step 1: Get valence contribution of the proposed new bond
    new_valence = vocab.bond_idx_to_valence(new_edge)
    if new_valence == -1:
        return False  # Invalid bond type

    # Step 2: Convert atom indices to chemical symbols
    atom1 = vocab.idx_to_atom(aid1)  # e.g., 'C', 'N', 'O', etc.
    atom2 = vocab.idx_to_atom(aid2)

    # Step 3: Calculate current valence usage for each atom
    # Sum up valence contributions from all existing bonds
    a1_current_valence = sum(list(map(vocab.bond_idx_to_valence, edges1)))
    a2_current_valence = sum(list(map(vocab.bond_idx_to_valence, edges2)))

    # Step 4: Special case handling for sulfur
    # Sulfur can have valence 2 (like in H2S) or 6 (like in SF6)
    # If sulfur currently has valence 2, we don't allow expansion to valence 6
    # This prevents unrealistic sulfur chemistry in generated molecules
    if (atom1 == 'S' and a1_current_valence == 2) or (atom2 == 'S' and a2_current_valence == 2):
        return False  # Don't allow sulfur valence expansion from 2 to 6

    # Step 5: Check valence limits for both atoms
    # Formula: current_valence + new_bond_valence + |formal_charge| ≤ max_valence
    #
    # Formal charge consideration:
    # - Positive charge reduces available valence (fewer electrons to share)
    # - Negative charge increases available valence (more electrons to share)
    # - We use absolute value as a conservative approximation

    atom1_valid = (a1_current_valence + new_valence + abs(c1) <= MAX_VALENCE[atom1])
    atom2_valid = (a2_current_valence + new_valence + abs(c2) <= MAX_VALENCE[atom2])

    return atom1_valid and atom2_valid


def cnt_atom(smi):
    """
    Count the number of atoms in a SMILES string.

    Implementation Note:
    - Uses MAX_VALENCE dictionary as a lookup for valid atomic symbols
    - MAX_VALENCE contains mappings like {'C': 4, 'N': 3, 'O': 2, ...}
    - Simple character matching approach works for basic SMILES strings

    Args:
        smi (str): SMILES string representation of a molecular fragment

    Returns:
        int: Number of atoms found in the SMILES string
    """
    atom_count = 0

    # Iterate through each character in the SMILES string
    for character in smi:
        # Check if character represents a valid atomic symbol
        # MAX_VALENCE dictionary contains all supported element symbols
        if character in MAX_VALENCE:
            atom_count += 1

    return atom_count


def cycle_check(i, j, mol):
    """
    Check if adding a bond between atoms i and j would create a chemically valid cycle.

    Chemical Background:
    - 3-membered rings (cyclopropanes): Highly strained, rare in drugs
    - 4-membered rings (cyclobutanes): Very strained, uncommon
    - 5-membered rings (cyclopentanes): Common, low strain
    - 6-membered rings (cyclohexanes): Very common, chair conformation stable
    - 7+ membered rings: Less common in small molecules, more flexible

    Implementation:
    Uses shortest path calculation to determine if atoms i and j are already
    connected through the existing molecular graph. If they are connected,
    adding a direct bond would create a cycle of length (shortest_path + 1).

    Args:
        i (int): Index of first atom
        j (int): Index of second atom
        mol (Chem.RWMol): RDKit molecule object representing current molecular state

    Returns:
        bool: True if the bond addition would create a valid cycle or no cycle,
              False if it would create an invalid cycle (3-4 or 7+ atoms)
    """

    # Calculate shortest path length between atoms i and j in current molecule
    # This represents the minimum number of bonds connecting the two atoms
    cycle_length = shortest_path_len(i, j, mol)

    # Determine if the proposed bond addition would create a valid cycle
    return cycle_length is None or (cycle_length > 4 and cycle_length < 7)


def shortest_path_len(i, j, mol):
    """
    Calculate the shortest path length between two atoms in a molecule using BFS.

    Chemical Background:
    - Molecular graphs represent atoms as nodes and bonds as edges
    - Path length = number of bonds in the shortest connection between atoms
    - Used for cycle detection: if atoms i and j are connected by path length n,
      adding a direct bond creates a cycle of length (n + 1)
    - Essential for validating chemically reasonable ring sizes

    Args:
        i (int): Index of the first atom (source)
        j (int): Index of the second atom (target)
        mol (Chem.RWMol): RDKit molecule object containing the molecular graph

    Returns:
        int or None: Shortest path length between atoms i and j
                    Returns None if atoms are not connected

    Time Complexity: O(V + E) where V = atoms, E = bonds
    Space Complexity: O(V) for queue and visited set
    """

    queue = Queue()
    visited = {}

    # Start BFS from atom i with initial distance of 1
    # Distance represents the number of bonds in the path
    start_atom = mol.GetAtomWithIdx(i)
    queue.put((start_atom, 1))
    visited[i] = True

    # BFS traversal to find shortest path
    while not queue.empty():
        current_atom, current_distance = queue.get()

        # Explore all neighboring atoms (bonded atoms)
        neighbor_indices = []
        for neighbor_atom in current_atom.GetNeighbors():
            neighbor_idx = neighbor_atom.GetIdx()

            # Check if we've reached the target atom
            if neighbor_idx == j:
                # Found target! Return total distance
                # current_distance represents bonds to current_atom
                # +1 for the bond from current_atom to target
                return current_distance + 1

            # Add unvisited neighbors to queue for next level exploration
            if neighbor_idx not in visited:
                visited[neighbor_idx] = True
                neighbor_indices.append(neighbor_idx)

                # Enqueue neighbor with incremented distance
                next_atom = mol.GetAtomWithIdx(neighbor_idx)
                queue.put((next_atom, current_distance + 1))

    # BFS completed without finding target atom j
    # This means atoms i and j are in disconnected components
    return None



