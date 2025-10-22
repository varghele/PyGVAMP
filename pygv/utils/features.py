def get_amino_acid_features(residue_name):
    """
    Convert amino acid to property-based features.
    Returns a compact feature vector based on amino acid properties.
    """
    # Amino acid properties (normalized values)
    aa_properties = {
        'ALA': [0.0, 0.0, 0.0, 0.0],  # [hydrophobic, polar, charged, aromatic]
        'ARG': [0.0, 1.0, 1.0, 0.0],  # Basic, charged
        'ASN': [0.0, 1.0, 0.0, 0.0],  # Polar
        'ASP': [0.0, 1.0, -1.0, 0.0],  # Acidic, charged
        'CYS': [0.5, 0.0, 0.0, 0.0],  # Slightly hydrophobic
        'GLN': [0.0, 1.0, 0.0, 0.0],  # Polar
        'GLU': [0.0, 1.0, -1.0, 0.0],  # Acidic, charged
        'GLY': [0.0, 0.0, 0.0, 0.0],  # Neutral
        'HIS': [0.0, 1.0, 0.5, 1.0],  # Polar, weakly charged, aromatic
        'ILE': [1.0, 0.0, 0.0, 0.0],  # Hydrophobic
        'LEU': [1.0, 0.0, 0.0, 0.0],  # Hydrophobic
        'LYS': [0.0, 1.0, 1.0, 0.0],  # Basic, charged
        'MET': [1.0, 0.0, 0.0, 0.0],  # Hydrophobic
        'PHE': [1.0, 0.0, 0.0, 1.0],  # Hydrophobic, aromatic
        'PRO': [0.5, 0.0, 0.0, 0.0],  # Special structure
        'SER': [0.0, 1.0, 0.0, 0.0],  # Polar
        'THR': [0.0, 1.0, 0.0, 0.0],  # Polar
        'TRP': [1.0, 0.0, 0.0, 1.0],  # Hydrophobic, aromatic
        'TYR': [0.5, 1.0, 0.0, 1.0],  # Aromatic, polar
        'VAL': [1.0, 0.0, 0.0, 0.0],  # Hydrophobic
    }

    return aa_properties.get(residue_name, [0.0, 0.0, 0.0, 0.0])


def get_amino_acid_labels(residue_name):
    """
    Convert amino acid to unique integer labels.
    Returns a unique integer label for each amino acid type.
    """
    # Amino acid to unique label mapping
    aa_labels = {
        'ALA': 0,  # Alanine
        'ARG': 1,  # Arginine
        'ASN': 2,  # Asparagine
        'ASP': 3,  # Aspartic acid
        'CYS': 4,  # Cysteine
        'GLN': 5,  # Glutamine
        'GLU': 6,  # Glutamic acid
        'GLY': 7,  # Glycine
        'HIS': 8,  # Histidine
        'ILE': 9,  # Isoleucine
        'LEU': 10,  # Leucine
        'LYS': 11,  # Lysine
        'MET': 12,  # Methionine
        'PHE': 13,  # Phenylalanine
        'PRO': 14,  # Proline
        'SER': 15,  # Serine
        'THR': 16,  # Threonine
        'TRP': 17,  # Tryptophan
        'TYR': 18,  # Tyrosine
        'VAL': 19,  # Valine
    }

    return aa_labels.get(residue_name, 20)  # Return 20 for unknown residues
