import mdtraj as md
import os


def load_trajectory_old(dcd_file: str, topology_file: str, stride: int = 1, atom_selection: str = None):
    """
    Load a molecular dynamics trajectory from DCD file.

    Parameters
    ----------
    dcd_file : str
        Path to the DCD trajectory file
    topology_file : str
        Path to the topology file (PDB, PSF, etc.)
    stride : int, optional
        Load every nth frame to reduce memory usage, default=1
    atom_selection : str, optional
        MDTraj selection string to filter specific atoms, default=None (all atoms)

    Returns
    -------
    mdtraj.Trajectory
        Loaded trajectory object
    """
    # Check if files exist
    if not os.path.exists(dcd_file):
        raise FileNotFoundError(f"Trajectory file not found: {dcd_file}")
    if not os.path.exists(topology_file):
        raise FileNotFoundError(f"Topology file not found: {topology_file}")

    print(f"Loading trajectory: {dcd_file}")
    print(f"Using topology: {topology_file}")
    print(f"Stride: {stride}")

    # Load trajectory
    try:
        traj = md.load(dcd_file, top=topology_file, stride=stride)
        print(f"Loaded trajectory with {traj.n_frames} frames and {traj.n_atoms} atoms")

        # Select specific atoms if requested
        if atom_selection:
            atom_indices = traj.topology.select(atom_selection)
            if len(atom_indices) == 0:
                raise ValueError(f"Selection '{atom_selection}' matched no atoms")

            traj = traj.atom_slice(atom_indices)
            print(f"Applied selection '{atom_selection}': {traj.n_atoms} atoms selected")

        # Print basic trajectory information
        timestep = None
        if traj.timestep:
            timestep = traj.timestep
            print(f"Trajectory timestep: {timestep} ps")
        elif hasattr(traj, 'time') and len(traj.time) > 1:
            timestep = traj.time[1] - traj.time[0]
            print(f"Trajectory timestep (inferred): {timestep} ps")
        else:
            print("Could not determine trajectory timestep")

        duration = traj.n_frames * stride * timestep if timestep else None
        if duration:
            print(f"Trajectory duration: {duration} ps ({duration / 1000:.3f} ns)")

        return traj

    except Exception as e:
        print(f"Error loading trajectory: {str(e)}")
        raise


def load_trajectory(dcd_file: str, topology_file: str, stride: int = 1, atom_selection: str = None):
    """
    Load a molecular dynamics trajectory from DCD file with detailed atom selection information.

    Parameters
    ----------
    dcd_file : str
        Path to the DCD trajectory file
    topology_file : str
        Path to the topology file (PDB, PSF, etc.)
    stride : int, optional
        Load every nth frame to reduce memory usage, default=1
    atom_selection : str, optional
        MDTraj selection string to filter specific atoms, default=None (all atoms)

    Returns
    -------
    mdtraj.Trajectory
        Loaded trajectory object
    """
    import mdtraj as md
    import os
    import numpy as np

    # Check if files exist
    if not os.path.exists(dcd_file):
        raise FileNotFoundError(f"Trajectory file not found: {dcd_file}")
    if not os.path.exists(topology_file):
        raise FileNotFoundError(f"Topology file not found: {topology_file}")

    print(f"Loading trajectory: {dcd_file}")
    print(f"Using topology: {topology_file}")
    print(f"Stride: {stride}")

    # First load just the topology to analyze atom selection
    try:
        topology = md.load_topology(topology_file)
        print(f"Topology loaded with {topology.n_atoms} atoms and {topology.n_residues} residues")

        # If atom selection is specified, show detailed info about selected atoms
        if atom_selection:
            # Get atom indices for the selection
            atom_indices = topology.select(atom_selection)

            if len(atom_indices) == 0:
                print(f"Warning: Selection '{atom_selection}' matched no atoms")
            else:
                print(f"\nSelection '{atom_selection}' matched {len(atom_indices)} atoms:")

                # Print information about selected atoms
                print("\nAtom Selection Details:")
                print("-" * 80)
                print(f"{'Index':<10}{'Atom':<8}{'Residue':<10}{'ResID':<8}{'Chain':<8}{'Element':<8}")
                print("-" * 80)

                # Print details for up to 20 atoms (to avoid excessive output)
                max_atoms_to_show = min(20, len(atom_indices))
                for i, atom_idx in enumerate(atom_indices[:max_atoms_to_show]):
                    atom = topology.atom(atom_idx)
                    print(f"{atom_idx:<10}{atom.name:<8}{atom.residue.name:<10}{atom.residue.resSeq:<8}"
                          f"{atom.residue.chain.index:<8}{atom.element.symbol if atom.element else 'None':<8}")

                # If there are more atoms than shown, indicate this
                if len(atom_indices) > max_atoms_to_show:
                    print(f"... and {len(atom_indices) - max_atoms_to_show} more atoms")

                # Show residue distribution
                residue_counts = {}
                for atom_idx in atom_indices:
                    atom = topology.atom(atom_idx)
                    res_name = atom.residue.name
                    residue_counts[res_name] = residue_counts.get(res_name, 0) + 1

                print("\nResidue Distribution:")
                for res_name, count in sorted(residue_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {res_name}: {count} atoms")

        # Detailed analysis of protein structure if it's a protein
        protein_atoms = topology.select("protein")
        if len(protein_atoms) > 0:
            print("\nProtein Structure Analysis:")
            ca_atoms = topology.select("name CA and protein")
            print(f"  Protein atoms: {len(protein_atoms)}")
            print(f"  CA atoms: {len(ca_atoms)}")
            print(f"  Residues with CA atoms: {len(ca_atoms)} (should match number of residues in protein)")

            # Show chains and residues per chain
            chains = {}
            for atom_idx in ca_atoms:
                atom = topology.atom(atom_idx)
                chain_id = atom.residue.chain.index
                chains[chain_id] = chains.get(chain_id, 0) + 1

            print("\nChain Distribution (CA atoms):")
            for chain_id, count in sorted(chains.items()):
                print(f"  Chain {chain_id}: {count} residues")

    except Exception as e:
        print(f"Error analyzing topology: {str(e)}")

    # Now load the actual trajectory
    try:
        traj = md.load(dcd_file, top=topology_file, stride=stride)
        print(f"\nLoaded trajectory with {traj.n_frames} frames and {traj.n_atoms} atoms")

        # Select specific atoms if requested
        if atom_selection:
            atom_indices = traj.topology.select(atom_selection)
            if len(atom_indices) == 0:
                raise ValueError(f"Selection '{atom_selection}' matched no atoms")

            traj = traj.atom_slice(atom_indices)
            print(f"Applied selection '{atom_selection}': {traj.n_atoms} atoms selected")

        # Print basic trajectory information
        timestep = None
        if hasattr(traj, 'timestep') and traj.timestep is not None:
            timestep = traj.timestep
            print(f"Trajectory timestep: {timestep} ps")
        elif hasattr(traj, 'time') and len(traj.time) > 1:
            timestep = traj.time[1] - traj.time[0]
            print(f"Trajectory timestep (inferred): {timestep} ps")
        else:
            print("Could not determine trajectory timestep")

        duration = traj.n_frames * stride * timestep if timestep else None
        if duration:
            print(f"Trajectory duration: {duration} ps ({duration / 1000:.3f} ns)")

        # Calculate basic descriptive statistics for trajectory
        if traj.n_frames > 0:
            print("\nTrajectory Statistics:")

            # Calculate RMSD to first frame
            rmsd = md.rmsd(traj, traj, frame=0)
            print(f"  RMSD range to first frame: {rmsd.min():.3f} - {rmsd.max():.3f} nm")

            # Calculate pairwise RMSD between frames
            if traj.n_frames > 1 and traj.n_frames <= 100:  # Only for reasonably sized trajectories
                all_rmsd = np.zeros((traj.n_frames, traj.n_frames))
                for i in range(traj.n_frames):
                    all_rmsd[i] = md.rmsd(traj, traj, frame=i)

                print(f"  Mean pairwise RMSD: {all_rmsd.mean():.3f} nm")
                print(f"  Max pairwise RMSD: {all_rmsd.max():.3f} nm")

            # Calculate radius of gyration for first, middle and last frames
            try:
                rg_first = md.compute_rg(traj[0])[0]
                rg_last = md.compute_rg(traj[-1])[0]
                rg_middle = md.compute_rg(traj[traj.n_frames // 2])[0]
                print(f"  Radius of gyration (first frame): {rg_first:.3f} nm")
                print(f"  Radius of gyration (middle frame): {rg_middle:.3f} nm")
                print(f"  Radius of gyration (last frame): {rg_last:.3f} nm")
            except:
                print("  Could not calculate radius of gyration")

        return traj

    except Exception as e:
        print(f"Error loading trajectory: {str(e)}")
        raise

# Load a full trajectory
traj = load_trajectory(
    dcd_file=os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/2JOF-0-protein-000.dcd'),
    topology_file=os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/2JOF-0-protein.pdb'),
    stride=10  # Load every 10th frame to reduce memory usage
)

# Load a trajectory with only C-alpha atoms
#ca_traj = load_trajectory(
#    dcd_file=os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/TRP/DESRES-Trajectory_2JOF-0-c-alpha/2JOF-0-c-alpha/2JOF-0-c-alpha-000.dcd'),
#    topology_file=os.path.expanduser('~/PycharmProjects/DDVAMP/datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/2JOF-0-protein.pdb'),
#    stride=10,
#    atom_selection="name CA"  # Select only alpha carbon atoms
#)

# Basic analysis examples
print(f"Trajectory shape: {traj.xyz.shape}")  # (n_frames, n_atoms, 3)

# Calculate RMSD to first frame
rmsd = md.rmsd(traj, traj, frame=0)
print(f"RMSD range: {rmsd.min():.3f} - {rmsd.max():.3f} nm")

# Superpose trajectory on first frame
aligned_traj = traj.superpose(traj, 0)

# Save a specific frame as PDB
frame_idx = 100
#traj[frame_idx].save_pdb(f"frame_{frame_idx}.pdb")