import torch
import numpy as np
from psevo.evo.generator import PSVAEMoleculeGenerator, EvolutionaryMoleculeOptimizer


def main():
    """Run evolutionary molecule optimization."""

    # Initialize generator
    print("Loading PS-VAE generator...")
    generator = PSVAEMoleculeGenerator(
        model_path="checkpoints/best_model_ml3_zinc250.pt",
        tokenizer_path="vocab_zinc250",
        device="cuda"
    )

    # Add docking targets
    print("Loading docking targets...")
    generator.add_docking_target("DRD2", "DRD2")

    # Initialize evolutionary optimizer
    print("Setting up evolutionary optimizer...")
    optimizer = EvolutionaryMoleculeOptimizer(
        generator=generator,
        latent_dim=56,
        population_size=200
    )

    # Run evolution
    print("Starting evolutionary optimization...")
    final_population, final_fitness = optimizer.evolve(
        generations=200,
        mutation_rate=0.1
    )

    # Generate final molecules from best latent vectors
    print("\nGenerating final molecules...")
    best_indices = np.argsort(final_fitness)[-5:]  # Top 5
    best_latents = final_population[best_indices]

    best_molecules = generator.generate_molecules(best_latents)

    print("\nTop 5 optimized molecules:")
    for i, (mol, fitness) in enumerate(zip(best_molecules, np.array(final_fitness)[best_indices])):
        print(f"  {i + 1}: {mol} (fitness: {fitness:.3f})")

    print(f"\nEvolution completed!")
    print(f"Best fitness: {max(final_fitness):.3f}")
    print(f"Average fitness: {np.mean(final_fitness):.3f}")

if __name__ == "__main__":
    main()