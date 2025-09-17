import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED

import os
import sys
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
sys.path.remove(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

#sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))

#sys.path.append("C:\\Users\\varghele\\miniconda3\\pkgs\\rdkit-2024.09.5-py311h80b0796_0\\Library\\share\\RDKit\\Contrib\\SA_Score")
#import sascorer

def get_qed(molecule):
    """ Calculate QED (Quantitative Estimate of Drug-likeness) for a molecule.
    QED is a quantitative measure of drug-likeness that combines multiple
    molecular descriptors into a single score. It was developed by Bickerton
    et al. and is based on the concept of desirability functions applied to
    eight molecular properties commonly used in medicinal chemistry.

    The QED score ranges from 0 to 1, where:
    - Higher values (closer to 1) indicate better drug-likeness
    - Lower values (closer to 0) indicate poor drug-like properties

    QED considers the following molecular descriptors:
    - Molecular weight (MW)
    - Octanol-water partition coefficient (LogP)
    - Number of hydrogen bond donors (HBD)
    - Number of hydrogen bond acceptors (HBA)
    - Polar surface area (PSA)
    - Number of rotatable bonds (ROTB)
    - Number of aromatic rings (AROM)
    - Number of structural alerts (ALERTS)

    Args:
        molecule (Chem.Mol): RDKit molecule object

    Returns:
        float: QED score in range [0, 1], where higher values indicate
               better drug-likeness
    """
    return QED.qed(molecule)


def get_sa(molecule):
    """ Calculate Synthetic Accessibility (SA) score for a molecule.
    The SA score estimates how difficult it would be to synthesize a given
    molecule based on the complexity of its structure and the availability
    of starting materials. The score is computed using the SAScore algorithm
    which analyzes molecular fragments and their frequencies in chemical
    databases.

    The SA score ranges from 1 to 10, where:
    - Lower values (closer to 1) indicate easier synthesis
    - Higher values (closer to 10) indicate more difficult synthesis

    This scoring system helps medicinal chemists prioritize compounds
    that are not only biologically active but also synthetically accessible.

    Args:
        molecule (Chem.Mol): RDKit molecule object

    Returns:
        float: SA score in range [1, 10], where lower values indicate
               easier synthetic accessibility

    Note:
        This function is a wrapper around the SAScore implementation.
        The original SA score has the interpretation "lower = easier",
        which is often inverted in optimization contexts to "higher = better".
    """
    return sascorer.calculateScore(molecule)




def get_penalized_logp(mol):
    """ Calculate penalized LogP score for drug-likeness evaluation.
    This function computes a composite score that combines lipophilicity (LogP),
    synthetic accessibility (SA), and cycle penalty. The score is normalized
    based on statistics from the 250k_rndm_zinc_drugs_clean.smi dataset.

    Components:
    - LogP: Lipophilicity (partition coefficient between octanol and water)
    - SA: Synthetic accessibility score (inverted, so higher = easier to synthesize)
    - Cycle penalty: Penalizes large rings (>6 atoms) which are synthetically challenging

    The final score is a sum of normalized components, where higher values
    indicate better drug-like properties.

    Args:
        mol (Chem.Mol): RDKit molecule object

    Returns:
        float: Penalized LogP score (normalized, higher = better)

    Reference:
        Kusner et al. 2017 - "Grammar Variational Autoencoder"
        Normalization based on 250k_rndm_zinc_drugs_clean.smi dataset statistics
    """
    # Normalization constants from 250k_rndm_zinc_drugs_clean.smi dataset
    logp_mean = 2.4570953396190123
    logp_std = 1.434324401111988
    sa_mean = -3.0525811293166134
    sa_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    # Calculate LogP (lipophilicity)
    log_p = Descriptors.MolLogP(mol)

    # Calculate synthetic accessibility score (inverted: higher = easier)
    sa_score = -sascorer.calculateScore(mol)

    # Calculate cycle penalty for large rings
    # Find all cycles in the molecule using NetworkX
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))

    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        # Find the largest cycle
        cycle_length = max([len(cycle) for cycle in cycle_list])

    # Penalize cycles larger than 6 atoms (common in drug-like molecules)
    if cycle_length <= 6:
        cycle_penalty = 0
    else:
        cycle_penalty = cycle_length - 6

    cycle_score = -cycle_penalty  # Negative because larger cycles are worse

    # Normalize all components using dataset statistics
    normalized_logp = (log_p - logp_mean) / logp_std
    normalized_sa = (sa_score - sa_mean) / sa_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    # Return sum of normalized components
    return normalized_logp + normalized_sa + normalized_cycle


def get_normalized_property_scores(mol):
    """
    Calculate normalized molecular property scores for drug-likeness evaluation.

    This function computes three key molecular properties commonly used in
    drug discovery and normalizes them to a 0-1 scale where higher values
    indicate better drug-like characteristics.

    Properties calculated:
    - QED (Quantitative Estimate of Drug-likeness): Measures overall drug-likeness
    - SA (Synthetic Accessibility): Measures how easy a molecule is to synthesize
    - LogP (Penalized LogP): Measures lipophilicity with size penalty

    Args:
        mol (Chem.Mol): RDKit molecule object

    Returns:
        list: Three normalized property scores [qed_norm, sa_norm, logp_norm]
              All values are scaled to 0-1 range where higher = better
    """
    # Calculate raw property values
    qed = get_qed(mol)  # QED: already in 0-1 range, higher = better
    sa = get_sa(mol)  # SA: 1-10 scale, lower = better (easier to synthesize)
    logp = get_penalized_logp(mol)  # LogP: can be negative, optimal range around 0-3

    # Normalize properties to 0-1 scale with "higher is better" orientation
    qed_normalized = qed  # QED already 0-1, higher = better
    sa_normalized = 1 - sa / 10  # Invert SA: 1-(1/10)=0.9 (easy), 1-(10/10)=0 (hard)
    logp_normalized = (logp + 10) / 13  # Shift and scale: (-10+10)/13=0, (3+10)/13=1

    return [qed_normalized, sa_normalized, logp_normalized]
