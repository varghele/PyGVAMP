import networkx as nx
import hashlib
from torch_geometric.data import Data
import torch
from typing import List, Dict


# Original WeisfeilerLehmanMachine (from the paper)
class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """

    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()


# Your feature extractor (simplified version for comparison)
def your_wl_subgraphs(adj_list: List[List[int]],
                      initial_features: Dict[int, str],
                      num_nodes: int,
                      max_degree: int = 2) -> List[str]:
    """Your WL implementation from graph2vec_v8.txt"""
    all_subgraphs = []
    current_features = initial_features.copy()

    # Add degree 0 subgraphs (just node labels)
    for node in range(num_nodes):
        all_subgraphs.append(current_features[node])

    # Iterative WL relabeling
    for iteration in range(max_degree):
        new_features = {}

        for node in range(num_nodes):
            # Get neighbor features
            neighbor_features = [current_features[neighbor] for neighbor in adj_list[node]]

            # Create feature string (matching original format)
            features = [str(current_features[node])] + sorted([str(feat) for feat in neighbor_features])
            features_str = "_".join(features)  # Use underscore separator like original

            # Use MD5 hash like original (deterministic)
            hash_object = hashlib.md5(features_str.encode())
            new_features[node] = hash_object.hexdigest()

        # Add all subgraphs from this iteration
        for node in range(num_nodes):
            all_subgraphs.append(new_features[node])

        current_features = new_features

    return all_subgraphs


def create_test_graphs():
    """Create test graphs for comparison."""
    graphs = []

    # Triangle
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2), (2, 0)])
    graphs.append(("Triangle", G1))

    # Path
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2)])
    graphs.append(("Path", G2))

    # Star
    G3 = nx.Graph()
    G3.add_edges_from([(0, 1), (0, 2), (0, 3)])
    graphs.append(("Star", G3))

    # Square
    G4 = nx.Graph()
    G4.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    graphs.append(("Square", G4))

    # Pentagon
    G5 = nx.Graph()
    G5.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    graphs.append(("Pentagon", G5))

    # Complete graph K4
    G6 = nx.Graph()
    G6.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    graphs.append(("Complete K4", G6))

    # Tree (binary tree)
    G7 = nx.Graph()
    G7.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    graphs.append(("Binary Tree", G7))

    return graphs


def networkx_to_adjacency_list(G):
    """Convert NetworkX graph to adjacency list format."""
    num_nodes = len(G.nodes())
    adj_list = [[] for _ in range(num_nodes)]

    for u, v in G.edges():
        adj_list[u].append(v)
        adj_list[v].append(u)

    return adj_list, num_nodes


def compare_wl_implementations_extended():
    """Compare original WeisfeilerLehmanMachine with your implementation for multiple iterations."""
    print("Comparing WeisfeilerLehmanMachine implementations with extended iterations...")
    print("=" * 80)

    test_graphs = create_test_graphs()
    max_iterations = 10  # Test up to 10 iterations

    for graph_name, G in test_graphs:
        print(f"\n{'=' * 20} Testing {graph_name} graph {'=' * 20}")
        print(f"Nodes: {list(G.nodes())}")
        print(f"Edges: {list(G.edges())}")

        # Prepare features (using degree as initial features)
        features = dict(nx.degree(G))
        features = {int(k): int(v) for k, v in features.items()}

        print(f"Initial features (degrees): {features}")

        # Test different numbers of iterations
        for iterations in range(1, max_iterations + 1):
            print(f"\n--- Testing with {iterations} iterations ---")

            # Original implementation
            original_wl = WeisfeilerLehmanMachine(G, features.copy(), iterations)
            original_features = original_wl.extracted_features

            # Your implementation
            adj_list, num_nodes = networkx_to_adjacency_list(G)
            initial_features_str = {node: str(features[node]) for node in range(num_nodes)}
            your_features = your_wl_subgraphs(adj_list, initial_features_str, num_nodes, iterations)

            # Compare results
            print(f"Original WL features ({len(original_features)}): {[f[:8] + '...' for f in original_features]}")
            print(f"Your WL features ({len(your_features)}): {[f[:8] + '...' for f in your_features]}")

            # Check if they match
            if original_features == your_features:
                print("✅ MATCH: Both implementations produce identical results!")
            else:
                print("❌ MISMATCH: Implementations produce different results!")

                # Show first few differences
                print("\nFirst few differences:")
                max_len = max(len(original_features), len(your_features))
                diff_count = 0
                for i in range(min(10, max_len)):  # Show first 10 differences
                    orig = original_features[i] if i < len(original_features) else "MISSING"
                    yours = your_features[i] if i < len(your_features) else "MISSING"
                    if orig != yours:
                        match = "❌"
                        diff_count += 1
                        print(f"  {i:2d}: {match} Original: {orig[:15]}... | Yours: {yours[:15]}...")
                    elif i < 5:  # Show first few matches too
                        print(f"  {i:2d}: ✅ Both: {orig[:15]}...")

                if diff_count > 10:
                    print(f"  ... and {diff_count - 10} more differences")

                # Stop testing higher iterations for this graph if mismatch found
                break

        print("-" * 60)


def analyze_iteration_patterns():
    """Analyze how features evolve with iterations."""
    print("\n" + "=" * 80)
    print("Analyzing iteration patterns...")

    # Use triangle for detailed analysis
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    features = {0: 2, 1: 2, 2: 2}  # All nodes have degree 2

    print(f"\nDetailed analysis of triangle graph with {len(G.nodes())} nodes:")
    print(f"Initial features: {features}")

    # Test up to 5 iterations with detailed output
    for iterations in range(1, 6):
        print(f"\n--- Iteration {iterations} Analysis ---")

        # Original approach
        original_wl = WeisfeilerLehmanMachine(G, features.copy(), iterations)
        original_features = original_wl.extracted_features

        # Your approach
        adj_list, num_nodes = networkx_to_adjacency_list(G)
        initial_features_str = {node: str(features[node]) for node in range(num_nodes)}
        your_features = your_wl_subgraphs(adj_list, initial_features_str, num_nodes, iterations)

        print(f"Original features: {len(original_features)} total")
        print(f"Your features: {len(your_features)} total")

        # Show features by iteration
        expected_per_iteration = num_nodes
        print(f"Expected features per iteration: {expected_per_iteration}")

        for iter_num in range(iterations + 1):  # +1 because we include degree 0
            start_idx = iter_num * expected_per_iteration
            end_idx = (iter_num + 1) * expected_per_iteration

            if start_idx < len(original_features):
                orig_iter = original_features[start_idx:end_idx]
                your_iter = your_features[start_idx:end_idx] if start_idx < len(your_features) else []

                print(f"  Iter {iter_num}: Original={[f[:8] + '...' for f in orig_iter]}")
                print(f"  Iter {iter_num}: Yours   ={[f[:8] + '...' for f in your_iter]}")
                print(f"  Iter {iter_num}: Match   ={orig_iter == your_iter}")


def test_convergence_behavior():
    """Test when WL features converge (stop changing)."""
    print("\n" + "=" * 80)
    print("Testing convergence behavior...")

    test_graphs = [
        ("Triangle", nx.Graph([(0, 1), (1, 2), (2, 0)])),
        ("Square", nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])),
        ("Complete K4", nx.complete_graph(4))
    ]

    for graph_name, G in test_graphs:
        print(f"\n--- Convergence analysis for {graph_name} ---")
        features = dict(nx.degree(G))
        features = {int(k): int(v) for k, v in features.items()}

        prev_features = None
        converged_at = None

        for iterations in range(1, 15):  # Test up to 15 iterations
            original_wl = WeisfeilerLehmanMachine(G, features.copy(), iterations)
            current_features = original_wl.extracted_features

            # Check if features have converged (last iteration same as previous)
            num_nodes = len(G.nodes())
            if iterations > 1:
                # Compare last iteration with previous last iteration
                last_iter_start = (iterations - 1) * num_nodes + num_nodes  # Skip degree 0
                prev_iter_start = (iterations - 2) * num_nodes + num_nodes

                if (last_iter_start < len(current_features) and
                        prev_iter_start < len(prev_features)):

                    last_iter = current_features[last_iter_start:last_iter_start + num_nodes]
                    prev_iter = prev_features[prev_iter_start:prev_iter_start + num_nodes]

                    if last_iter == prev_iter and converged_at is None:
                        converged_at = iterations - 1
                        print(f"  Converged at iteration {converged_at}")
                        break

            prev_features = current_features.copy()

            if iterations <= 5 or iterations % 5 == 0:  # Show progress
                print(f"  Iteration {iterations}: {len(current_features)} total features")

        if converged_at is None:
            print(f"  Did not converge within 15 iterations")


if __name__ == "__main__":
    compare_wl_implementations_extended()
    analyze_iteration_patterns()
    test_convergence_behavior()
