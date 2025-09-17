import torch
import time
from psevo.evo.generator import PSVAEMoleculeGenerator


def main():
    """Main function with comprehensive timing analysis."""

    print("Initializing PS-VAE Molecule Generator...")
    start_time = time.time()

    # Initialize generator
    generator = PSVAEMoleculeGenerator(
        model_path="checkpoints/best_model.pt",
        tokenizer_path="vocab_zinc250",
        device="cuda"
    )

    init_time = time.time() - start_time
    print(f"Generator initialization time: {init_time:.3f} seconds")

    # Add docking targets
    print("Loading docking targets...")
    target_start = time.time()
    generator.add_docking_target("DRD2", "DRD2")
    target_time = time.time() - target_start
    print(f"Docking target loading time: {target_time:.3f} seconds")

    # Test generation with timing
    num_molecules = 500
    print(f"\nGenerating {num_molecules} molecules...")

    test_latents = torch.randn(num_molecules, 56).to('cuda')

    # Time molecule generation only
    gen_start = time.time()
    molecules = generator.generate_molecules(test_latents)
    gen_time = time.time() - gen_start

    # Calculate generation statistics
    successful_generations = sum(1 for mol in molecules if mol is not None)
    failed_generations = num_molecules - successful_generations

    print(f"\nMolecule Generation Results:")
    print(f"  Total molecules: {num_molecules}")
    print(f"  Successful: {successful_generations}")
    print(f"  Failed: {failed_generations}")
    print(f"  Success rate: {successful_generations / num_molecules * 100:.1f}%")
    print(f"\nGeneration Timing:")
    print(f"  Total generation time: {gen_time:.3f} seconds")
    print(f"  Time per molecule: {gen_time / num_molecules * 1000:.2f} ms")
    print(f"  Time per successful molecule: {gen_time / successful_generations * 1000:.2f} ms")
    print(f"  Generation throughput: {successful_generations / gen_time:.1f} molecules/second")

    # Time fitness evaluation
    print(f"\nEvaluating fitness for all {num_molecules} molecules...")
    fitness_start = time.time()
    fitness_scores = generator.evaluate_fitness(test_latents)
    fitness_time = time.time() - fitness_start

    print(f"\nFitness Evaluation Timing:")
    print(f"  Total fitness time: {fitness_time:.3f} seconds")
    print(f"  Time per molecule: {fitness_time / num_molecules * 1000:.2f} ms")
    print(f"  Fitness throughput: {num_molecules / fitness_time:.1f} molecules/second")

    # Break down fitness timing components
    print(f"\nDetailed Timing Breakdown:")

    # Time just docking
    valid_molecules = [mol for mol in molecules if mol is not None]
    if valid_molecules:
        docking_start = time.time()
        docking_results = generator.calculate_docking_scores(valid_molecules)
        docking_time = time.time() - docking_start

        print(f"  Docking time: {docking_time:.3f} seconds")
        print(f"  Docking per molecule: {docking_time / len(valid_molecules) * 1000:.2f} ms")

        # Time complexity calculation
        complexity_start = time.time()
        complexity_scores = generator.calculate_complexity_scores(valid_molecules)
        complexity_time = time.time() - complexity_start

        print(f"  Complexity time: {complexity_time:.3f} seconds")
        print(f"  Complexity per molecule: {complexity_time / len(valid_molecules) * 1000:.2f} ms")

        # Time synthesizability calculation
        sa_start = time.time()
        sa_scores = generator.calculate_synthesizability_scores(valid_molecules)
        sa_time = time.time() - sa_start

        print(f"  Synthesizability time: {sa_time:.3f} seconds")
        print(f"  Synthesizability per molecule: {sa_time / len(valid_molecules) * 1000:.2f} ms")

    # Overall pipeline timing
    total_time = time.time() - start_time
    print(f"\nOverall Pipeline Timing:")
    print(f"  Total pipeline time: {total_time:.3f} seconds")
    print(f"  End-to-end per molecule: {total_time / num_molecules * 1000:.2f} ms")

    # Display sample results
    print(f"\nSample Results (first 10 molecules):")
    for i, (mol, fitness) in enumerate(zip(molecules[:10], fitness_scores[:10])):
        status = "✓" if mol is not None else "✗"
        mol_display = mol[:50] + "..." if mol and len(mol) > 50 else mol
        print(f"  {i + 1:2d} {status} {mol_display} (fitness: {fitness:.3f})")

    # Performance summary
    print(f"\n" + "=" * 60)
    print(f"PERFORMANCE SUMMARY")
    print(f"=" * 60)
    print(f"Generation Speed:     {successful_generations / gen_time:.1f} molecules/sec")
    print(f"Fitness Evaluation:   {num_molecules / fitness_time:.1f} molecules/sec")
    print(f"End-to-End Pipeline:  {num_molecules / total_time:.1f} molecules/sec")
    print(f"Success Rate:         {successful_generations / num_molecules * 100:.1f}%")
    print(f"Average Fitness:      {sum(fitness_scores) / len(fitness_scores):.3f}")


if __name__ == "__main__":
    main()

#import torch
#from psevo.evo.generator import PSVAEMoleculeGenerator

# Initialize generator
#generator = PSVAEMoleculeGenerator(model_path="checkpoints/best_model.pt",
#                                   tokenizer_path="vocab_zinc250",
#                                   device="cuda")

# Add docking targets using correct API
#generator.add_docking_target("DRD2", "DRD2")
#generator.add_docking_target("HTR2A", "HTR2A")

# Test generation
#test_latents = torch.randn(500, 56).to('cuda')
#molecules = generator.generate_molecules(test_latents)
#fitness_scores = generator.evaluate_fitness(test_latents)

#print("Generated molecules:")
#for i, (mol, fitness) in enumerate(zip(molecules, fitness_scores)):
#    print(f"  {i}: {mol} (fitness: {fitness:.3f})")