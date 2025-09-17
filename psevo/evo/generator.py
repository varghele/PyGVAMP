import torch
import numpy as np
from typing import List, Optional, Union
from dockstring import load_target

from psevo.tokenizer.mol_bpe_old import Tokenizer
from psevo.models.psvae import PSVAEModel

class PSVAEMoleculeGenerator:
    """ Molecule generator using PS-VAE for evolutionary algorithms.

    This class loads a trained PS-VAE model and generates molecules from
    latent encodings. It's designed to work with evolutionary algorithms
    where latent vectors are evolved and molecules are generated for
    fitness evaluation.

    Key Features:
    - Batch molecule generation from latent vectors
    - Integration with DockString for docking scores
    - Extensible fitness evaluation (docking + complexity + synthesizability)
    - Error handling for invalid molecules
    """

    def __init__(self, model_path: str, tokenizer_path: str, device: str = 'cuda'):
        """
        Initialize the molecule generator.

        Args:
            model_path (str): Path to trained PS-VAE model checkpoint
            tokenizer_path (str): Path to tokenizer
            device (str): Device for model inference
        """
        self.device = device

        # Load tokenizer
        self.tokenizer = Tokenizer(tokenizer_path)

        # Load trained model
        self.model = self._load_model(model_path)
        self.model.to(device)
        self.model.eval()

        # Generation parameters
        self.max_atom_num = 50
        self.add_edge_th = 0.5
        self.temperature = 1.0

        # Docking targets (can be extended)
        self.docking_targets = {}

    def _load_model(self, model_path: str):
        """Load the trained PS-VAE model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']

        # Initialize model
        model = PSVAEModel(config, self.tokenizer).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def add_docking_target(self, target_name: str, target_id: str):
        """
        Add a docking target for fitness evaluation.

        Args:
            target_name (str): Name for the target (e.g., 'DRD2')
            target_id (str): DockString target identifier
        """
        try:
            target = load_target(target_id)
            self.docking_targets[target_name] = target
            print(f"Added docking target: {target_name}")
        except Exception as e:
            print(f"Failed to load target {target_name}: {e}")

    def generate_molecules(self, latent_vectors: torch.Tensor) -> List[Optional[str]]:
        """
        Generate molecules from latent vectors.

        Args:
            latent_vectors (torch.Tensor): Batch of latent vectors [batch_size, latent_dim]

        Returns:
            List[Optional[str]]: List of SMILES strings (None for failed generations)
        """
        molecules = []

        with torch.no_grad():
            for latent_vec in latent_vectors:
                try:
                    # Generate molecule using PS-VAE
                    mol = self.model.generate_molecule_no_vae(
                        latent_vec,
                        self.max_atom_num,
                        self.add_edge_th,
                        self.temperature
                    )

                    # Convert to SMILES
                    if mol is not None:
                        from rdkit import Chem
                        smiles = Chem.MolToSmiles(mol)
                        molecules.append(smiles)
                    else:
                        molecules.append(None)

                except Exception as e:
                    print(f"Generation failed: {e}")
                    molecules.append(None)

        return molecules

    def calculate_docking_scores(self, smiles_list: List[str]) -> dict:
        """
        Calculate docking scores for molecules against all targets.

        Args:
            smiles_list (List[str]): List of SMILES strings

        Returns:
            dict: Docking scores {target_name: [scores]}
        """
        docking_results = {target_name: [] for target_name in self.docking_targets}

        for smiles in smiles_list:
            if smiles is None:
                # Add penalty scores for failed molecules
                for target_name in self.docking_targets:
                    docking_results[target_name].append(-50.0)
                continue

            for target_name, target in self.docking_targets.items():
                try:
                    # Correct DockString API usage
                    score, aux = target.dock(smiles)
                    docking_results[target_name].append(score)
                except Exception as e:
                    print(f"Docking failed for {smiles} against {target_name}: {e}")
                    # Penalize molecules that fail docking
                    docking_results[target_name].append(-50.0)

        return docking_results

    def calculate_complexity_scores(self, smiles_list: List[str]) -> List[float]:
        """
        Calculate molecular complexity scores.

        Args:
            smiles_list (List[str]): List of SMILES strings

        Returns:
            List[float]: Complexity scores (higher = more complex)
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        complexity_scores = []

        for smiles in smiles_list:
            if smiles is None:
                complexity_scores.append(0.0)
                continue

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Use molecular weight as simple complexity measure
                    # Can be extended with more sophisticated measures
                    complexity = Descriptors.MolWt(mol) / 500.0  # Normalize
                    complexity_scores.append(min(complexity, 1.0))
                else:
                    complexity_scores.append(0.0)
            except:
                complexity_scores.append(0.0)

        return complexity_scores

    def calculate_synthesizability_scores(self, smiles_list: List[str]) -> List[float]:
        """
        Calculate synthetic accessibility scores.

        Args:
            smiles_list (List[str]): List of SMILES strings

        Returns:
            List[float]: SA scores (higher = easier to synthesize)
        """
        try:
            import sascorer
        except ImportError:
            print("SAScore not available, returning default scores")
            return [0.5] * len(smiles_list)

        from rdkit import Chem

        sa_scores = []

        for smiles in smiles_list:
            if smiles is None:
                sa_scores.append(0.0)
                continue

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    sa_score = sascorer.calculateScore(mol)
                    # Convert to 0-1 scale where higher = easier
                    normalized_sa = 1.0 - (sa_score - 1.0) / 9.0
                    sa_scores.append(max(0.0, min(1.0, normalized_sa)))
                else:
                    sa_scores.append(0.0)
            except:
                sa_scores.append(0.0)

        return sa_scores

    def evaluate_fitness(self, latent_vectors: torch.Tensor,
                         weights: dict = None) -> List[float]:
        """
        Complete fitness evaluation for evolutionary algorithm.

        Args:
            latent_vectors (torch.Tensor): Batch of latent vectors
            weights (dict): Weights for different fitness components

        Returns:
            List[float]: Fitness scores for each latent vector
        """
        if weights is None:
            weights = {
                'docking': 1.0,
                'complexity': 0.1,
                'synthesizability': 0.2
            }

        # Generate molecules
        molecules = self.generate_molecules(latent_vectors)

        # Calculate all fitness components
        fitness_scores = []

        # Docking scores
        docking_results = self.calculate_docking_scores(molecules)

        # Other scores
        complexity_scores = self.calculate_complexity_scores(molecules)
        sa_scores = self.calculate_synthesizability_scores(molecules)

        # Combine scores
        for i in range(len(molecules)):
            fitness = 0.0

            # Docking component (use best docking score across targets)
            if self.docking_targets:
                best_docking = max([docking_results[target][i]
                                    for target in self.docking_targets])
                fitness += weights['docking'] * (-best_docking)  # Negative because lower is better

            # Complexity component
            fitness += weights['complexity'] * complexity_scores[i]

            # Synthesizability component
            fitness += weights['synthesizability'] * sa_scores[i]

            fitness_scores.append(fitness)

        return fitness_scores

    def set_generation_parameters(self, max_atom_num: int = 50,
                                  add_edge_th: float = 0.5,
                                  temperature: float = 1.0):
        """Set parameters for molecule generation."""
        self.max_atom_num = max_atom_num
        self.add_edge_th = add_edge_th
        self.temperature = temperature


class EvolutionaryMoleculeOptimizer:
    """ Simple evolutionary algorithm for molecule optimization.
    This class demonstrates how to use the PSVAEMoleculeGenerator
    in an evolutionary algorithm context with proper device handling.
    """

    def __init__(self, generator, latent_dim: int = 56, population_size: int = 100):
        self.generator = generator
        self.latent_dim = latent_dim
        self.population_size = population_size

        # Initialize random population on the same device as the generator
        device = next(generator.model.parameters()).device
        self.population = torch.randn(population_size, latent_dim, device=device)

    def evolve(self, generations: int = 50, mutation_rate: float = 0.1):
        """Run evolutionary optimization with proper device handling."""

        device = self.population.device

        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = self.generator.evaluate_fitness(self.population)

            # Selection (tournament selection)
            new_population = []
            for _ in range(self.population_size):
                # Tournament selection
                tournament_size = 5
                tournament_indices = np.random.choice(
                    self.population_size, tournament_size, replace=False
                )
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                new_population.append(self.population[winner_idx].clone())

            self.population = torch.stack(new_population)

            # Mutation - ensure tensors are on correct device
            mutation_mask = torch.rand(self.population_size, device=device) < mutation_rate
            mutation_noise = torch.randn_like(self.population) * 0.1
            self.population[mutation_mask] += mutation_noise[mutation_mask]

            # Report progress
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            print(f"Generation {generation}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}")

        return self.population, fitness_scores


if __name__ == "__main__":
    # Initialize generator
    generator = PSVAEMoleculeGenerator(model_path="checkpoints/best_model.pt",
                                       tokenizer_path="data/vocab.txt", device="cuda" )

    # Add docking targets using correct API
    generator.add_docking_target("DRD2", "DRD2")
    generator.add_docking_target("HTR2A", "HTR2A")

    # Test generation
    test_latents = torch.randn(5, 56)
    molecules = generator.generate_molecules(test_latents)
    fitness_scores = generator.evaluate_fitness(test_latents)

    print("Generated molecules:")
    for i, (mol, fitness) in enumerate(zip(molecules, fitness_scores)):
        print(f"  {i}: {mol} (fitness: {fitness:.3f})")



