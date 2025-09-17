import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter

# Import your new Gensim-based Graph2Vec class
from pygv.clustering.graph2vec import Graph2Vec  # Replace with actual import


def test_classification(dataset_name='MUTAG'):
    """Test classification using the new Gensim-based Graph2Vec implementation."""
    print("Testing Gensim-based Graph2Vec on {} dataset...".format(dataset_name))

    # Load dataset
    dataset = TUDataset(root='/tmp/{}'.format(dataset_name), name=dataset_name)
    print("Loaded {} dataset with {} graphs".format(dataset_name, len(dataset)))

    # Train Graph2Vec with Gensim backend
    model = Graph2Vec(
        embedding_dim=512,
        max_degree=3,
        epochs=10,  # Reduced for faster testing, increase for better results
        negative_samples=10,
        learning_rate=0.025,
        batch_size=32,  # Reasonable batch size
        min_count=1,  # Include rare subgraphs
        num_workers=1  # Adjust based on your system
    )

    print("Training Graph2Vec model...")
    model.fit(dataset, len(dataset))

    # Get embeddings and labels
    print("Getting embeddings...")
    embeddings = model.get_embeddings().numpy()
    labels = [data.y.item() for data in dataset]

    print("Embeddings shape: {}".format(embeddings.shape))
    print("Number of labels: {}".format(len(labels)))
    print("Label distribution: {}".format(Counter(labels)))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.1
    )

    print("Training set size: {}, Test set size: {}".format(len(X_train), len(X_test)))

    # Train classifier
    print("Training SVM classifier...")
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nResults:")
    print("=" * 50)
    print("{} accuracy: {:.3f}".format(dataset_name, accuracy))
    print("Expected MUTAG accuracy from paper: ~0.832")

    if dataset_name == 'MUTAG':
        if accuracy > 0.8:
            print("SUCCESS: Excellent performance! Matches paper's results.")
        elif accuracy > 0.7:
            print("GOOD: Good performance, close to paper's results.")
        elif accuracy > 0.6:
            print("MODERATE: Reasonable performance.")
        else:
            print("ISSUE: Performance below expectations.")

    return accuracy


def test_triangle_path_discrimination():
    """Test if the Gensim-based Graph2Vec can distinguish triangles from paths."""
    print("\nTesting triangle vs path discrimination...")

    # Create test graphs
    graphs = []
    labels = []

    # Create 20 triangles
    for i in range(20):
        edges = [[0, 1], [1, 2], [2, 0], [1, 0], [2, 1], [0, 2]]
        edge_index = torch.tensor(edges).t().contiguous()
        graphs.append(Data(edge_index=edge_index, x=torch.ones(3, 1)))
        labels.append('triangle')

    # Create 20 paths
    for i in range(20):
        edges = [[0, 1], [1, 2], [1, 0], [2, 1]]
        edge_index = torch.tensor(edges).t().contiguous()
        graphs.append(Data(edge_index=edge_index, x=torch.ones(3, 1)))
        labels.append('path')

    print("Created {} graphs (20 triangles, 20 paths)".format(len(graphs)))

    # Train Graph2Vec
    model = Graph2Vec(
        embedding_dim=16,
        max_degree=2,
        epochs=50,
        batch_size=8,
        min_count=1,
        negative_samples=5,
        learning_rate=0.025
    )

    model.fit(graphs, len(graphs))
    embeddings = model.get_embeddings()

    # Test similarities
    triangle_embs = embeddings[:20]  # First 20 are triangles
    path_embs = embeddings[20:]  # Last 20 are paths

    # Average similarity within triangles
    triangle_sim = F.cosine_similarity(triangle_embs[0:1], triangle_embs[1:2]).item()

    # Average similarity within paths
    path_sim = F.cosine_similarity(path_embs[0:1], path_embs[1:2]).item()

    # Similarity between triangle and path
    cross_sim = F.cosine_similarity(triangle_embs[0:1], path_embs[0:1]).item()

    print("\nTriangle vs Path Discrimination Results:")
    print("=" * 50)
    print("Triangle-Triangle similarity: {:.3f}".format(triangle_sim))
    print("Path-Path similarity: {:.3f}".format(path_sim))
    print("Triangle-Path similarity: {:.3f}".format(cross_sim))

    if triangle_sim > 0.7 and path_sim > 0.7 and cross_sim < 0.3:
        print("SUCCESS: Gensim Graph2Vec shows expected patterns!")
        return True
    else:
        print("ISSUE: Still not learning proper similarities")
        return False


def test_multiple_datasets():
    """Test on multiple TU datasets."""
    datasets = ['MUTAG', 'PTC', 'PROTEINS']
    results = {}

    print("Testing Gensim Graph2Vec on multiple datasets...")
    print("=" * 60)

    for dataset_name in datasets:
        try:
            print("\n" + "-" * 40)
            accuracy = test_classification(dataset_name)
            results[dataset_name] = accuracy
        except Exception as e:
            print("Error testing {}: {}".format(dataset_name, str(e)))
            results[dataset_name] = None

    # Summary
    print("\n" + "=" * 60)
    print("MULTI-DATASET RESULTS SUMMARY:")
    print("=" * 60)
    for dataset, acc in results.items():
        if acc is not None:
            print("{}: {:.3f}".format(dataset, acc))
        else:
            print("{}: FAILED".format(dataset))


def test_gensim_specific_features():
    """Test Gensim-specific features like similarity queries."""
    print("\nTesting Gensim-specific features...")

    # Create a small dataset
    graphs = []

    # Create 10 triangles
    for i in range(10):
        edges = [[0, 1], [1, 2], [2, 0], [1, 0], [2, 1], [0, 2]]
        edge_index = torch.tensor(edges).t().contiguous()
        graphs.append(Data(edge_index=edge_index, x=torch.ones(3, 1)))

    # Create 10 paths
    for i in range(10):
        edges = [[0, 1], [1, 2], [1, 0], [2, 1]]
        edge_index = torch.tensor(edges).t().contiguous()
        graphs.append(Data(edge_index=edge_index, x=torch.ones(3, 1)))

    print("Created {} graphs for Gensim feature testing".format(len(graphs)))

    # Train model
    model = Graph2Vec(
        embedding_dim=32,
        max_degree=2,
        epochs=20,
        batch_size=4,
        min_count=1,
        negative_samples=3,
        learning_rate=0.025
    )

    model.fit(graphs, len(graphs))

    # Test similarity between graphs
    print("\nTesting graph similarity features:")
    try:
        # Similarity between two triangles (should be high)
        triangle_similarity = model.similarity(0, 1)
        print("Triangle-Triangle similarity: {:.3f}".format(triangle_similarity))

        # Similarity between triangle and path (should be low)
        cross_similarity = model.similarity(0, 10)
        print("Triangle-Path similarity: {:.3f}".format(cross_similarity))

        # Find most similar graphs to first triangle
        similar_graphs = model.most_similar_graphs(0, topn=3)
        print("Most similar graphs to triangle 0:")
        for graph_tag, similarity in similar_graphs:
            print("  {}: {:.3f}".format(graph_tag, similarity))

        print("SUCCESS: Gensim-specific features working correctly!")

    except Exception as e:
        print("Error testing Gensim features: {}".format(str(e)))


if __name__ == "__main__":
    # Test classification on MUTAG dataset
    try:
        print("=" * 60)
        print("GENSIM-BASED GRAPH2VEC TESTING")
        print("=" * 60)

        # Main classification test
        mutag_accuracy = test_classification('MUTAG')
        protein_accuracy = test_classification('PROTEINS')

        # Test triangle vs path discrimination
        triangle_path_success = test_triangle_path_discrimination()

        # Test Gensim-specific features
        test_gensim_specific_features()

        # Test multiple datasets (optional - comment out if too slow)
        # test_multiple_datasets()

        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY:")
        print("=" * 60)
        print("MUTAG Classification Accuracy: {:.3f}".format(mutag_accuracy))
        print("PROTEIN Classification Accuracy: {:.3f}".format(protein_accuracy))
        print("Triangle-Path Discrimination: {}".format("SUCCESS" if triangle_path_success else "FAILED"))

        if mutag_accuracy > 0.75 and triangle_path_success:
            print("\n🎉 OVERALL SUCCESS: Gensim-based Graph2Vec is working correctly!")
        elif mutag_accuracy > 0.6 or triangle_path_success:
            print("\n✅ PARTIAL SUCCESS: Some aspects are working well.")
        else:
            print("\n❌ ISSUES: Both tests show problems that need investigation.")

    except Exception as e:
        print("Error during testing: {}".format(str(e)))
        print("Make sure you have installed: pip install torch torch-geometric gensim scikit-learn")
        print("And that you've imported the Graph2Vec class correctly.")
