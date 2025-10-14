import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm
import pickle
import os
import hashlib
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
import time
from gensim.models.callbacks import CallbackAny2Vec


class TrainingCallback(CallbackAny2Vec):
    """Callback to track Doc2Vec training progress with loss monitoring."""

    def __init__(self, compute_loss=True):
        self.epoch = 0
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.compute_loss = compute_loss
        self.losses = []  # Store loss history

    def on_epoch_begin(self, model):
        self.epoch_start_time = time.time()
        print(f"Epoch {self.epoch + 1} started...")

    def on_epoch_end(self, model):
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.start_time

        # Get current learning rate
        current_lr = model.alpha

        # Get current loss if available
        loss_str = ""
        if self.compute_loss and hasattr(model, 'get_latest_training_loss'):
            try:
                # Get the latest training loss
                current_loss = model.get_latest_training_loss()
                self.losses.append(current_loss)

                # Calculate loss change if we have previous losses
                if len(self.losses) > 1:
                    loss_change = current_loss - self.losses[-2]
                    loss_str = f", Loss: {current_loss:.6f} (Δ: {loss_change:+.6f})"
                else:
                    loss_str = f", Loss: {current_loss:.6f}"

            except Exception as e:
                # Fallback: try to access training loss directly
                if hasattr(model, 'running_training_loss'):
                    current_loss = model.running_training_loss
                    self.losses.append(current_loss)
                    loss_str = f", Loss: {current_loss:.6f}"
                else:
                    loss_str = ", Loss: N/A"

        print(f"Epoch {self.epoch + 1} completed in {epoch_time:.2f}s "
              f"(Total: {total_time:.2f}s, LR: {current_lr:.6f}{loss_str})")

        self.epoch += 1

    def on_train_begin(self, model):
        print(f"Starting Doc2Vec training with {model.epochs} epochs...")
        print(f"Vocabulary size: {len(model.wv)}")
        print(f"Vector size: {model.vector_size}")
        print(f"Loss computation: {'Enabled' if self.compute_loss else 'Disabled'}")

    def on_train_end(self, model):
        total_time = time.time() - self.start_time
        print(f"Doc2Vec training completed in {total_time:.2f}s")

        # Print loss summary if available
        if self.losses:
            print(f"Loss progression: {self.losses[0]:.6f} → {self.losses[-1]:.6f}")
            print(f"Total loss reduction: {self.losses[0] - self.losses[-1]:.6f}")


class EnhancedTrainingCallback(CallbackAny2Vec):
    """Enhanced callback with better loss tracking and model evaluation."""

    def __init__(self, compute_loss=True, eval_every=5):
        self.epoch = 0
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.compute_loss = compute_loss
        self.losses = []
        self.eval_every = eval_every

    def on_epoch_end(self, model):
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.start_time
        current_lr = model.alpha

        # Try multiple ways to get loss
        loss_str = ""
        current_loss = None

        # Method 1: get_latest_training_loss
        if hasattr(model, 'get_latest_training_loss'):
            try:
                current_loss = model.get_latest_training_loss()
            except:
                pass

        # Method 2: running_training_loss
        if current_loss==0.0 and hasattr(model, 'running_training_loss'):
            current_loss = model.running_training_loss

        # Method 3: Compute perplexity as proxy for loss
        if current_loss==0.0 and self.epoch % self.eval_every == 0:
            try:
                # Sample some documents and compute average log probability
                sample_docs = model.docvecs.doctags[:min(100, len(model.docvecs.doctags))]
                log_probs = []
                for doc_tag in sample_docs:
                    try:
                        # Get document vector
                        doc_vec = model.docvecs[doc_tag]
                        # This is a rough approximation
                        log_prob = np.mean([model.wv.similarity(word, doc_vec)
                                            for word in list(model.wv.index_to_key)[:10]])
                        log_probs.append(log_prob)
                    except:
                        continue

                if log_probs:
                    current_loss = -np.mean(log_probs)  # Negative log likelihood approximation
            except:
                pass

        if current_loss is not None:
            self.losses.append(current_loss)
            if len(self.losses) > 1:
                loss_change = current_loss - self.losses[-2]
                loss_str = f", Loss: {current_loss:.6f} (Δ: {loss_change:+.6f})"
            else:
                loss_str = f", Loss: {current_loss:.6f}"

        print(f"Epoch {self.epoch + 1} completed in {epoch_time:.2f}s "
              f"(Total: {total_time:.2f}s, LR: {current_lr:.6f}{loss_str})")

        self.epoch += 1


class Graph2Vec:
    """
    Graph2Vec implementation using Gensim's Doc2Vec with PyTorch Geometric graph processing.

    This class combines:
    - PyTorch Geometric for efficient graph data handling and WL subgraph extraction
    - Gensim's proven Doc2Vec (PV-DBOW) implementation for embedding learning
    """

    def __init__(self,
                 embedding_dim: int = 128,
                 max_degree: int = 2,
                 negative_samples: int = 5,
                 learning_rate: float = 0.025,
                 min_learning_rate: float = 0.0001,
                 epochs: int = 10,
                 min_count: int = 5,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 vocab_cache_path: Optional[str] = None,
                 window: int = 0,  # Doc2Vec parameter (0 for PV-DBOW)
                 dm: int = 0):  # Doc2Vec parameter (0 for PV-DBOW)
        """
        Initialize Graph2Vec with Gensim Doc2Vec backend.

        Args:
            embedding_dim: Dimension of graph embeddings
            max_degree: Maximum degree of rooted subgraphs to consider
            negative_samples: Number of negative samples for training
            learning_rate: Initial learning rate for optimization
            min_learning_rate: Minimum learning rate (for decay)
            epochs: Number of training epochs
            min_count: Minimum frequency for subgraph to be included in vocabulary
            batch_size: Batch size for processing graphs
            num_workers: Number of workers for DataLoader
            vocab_cache_path: Path to cache vocabulary
            window: Context window size (0 for PV-DBOW)
            dm: Training algorithm (0=PV-DBOW, 1=PV-DM)
        """
        self.embedding_dim = embedding_dim
        self.max_degree = max_degree
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.epochs = epochs
        self.min_count = min_count
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_cache_path = vocab_cache_path
        self.window = window
        self.dm = dm

        # Will be initialized after vocabulary creation
        self.subgraph_vocab = {}
        self.vocab_size = 0
        self.doc2vec_model = None

        # For caching
        self.cached_subgraphs = {}
        self.graph_documents = []

    def fit(self, dataset, num_graphs: int) -> 'Graph2Vec':
        """
        Train Graph2Vec using Gensim's Doc2Vec on PyTorch Geometric graphs.

        Args:
            dataset: PyTorch Geometric dataset
            num_graphs: Number of graphs in dataset

        Returns:
            Self for method chaining
        """
        print("Extracting subgraphs and building vocabulary...")

        # Build vocabulary and extract subgraphs using PyTorch Geometric
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.subgraph_vocab = self._build_vocabulary(dataloader)
        self.vocab_size = len(self.subgraph_vocab)

        if self.vocab_size == 0:
            raise ValueError("No subgraphs found in vocabulary")

        # Cache all subgraphs
        print("Pre-extracting and caching subgraphs...")
        self.cached_subgraphs = self._cache_all_subgraphs(dataset, num_graphs)

        # Convert graphs to Doc2Vec documents
        print("Converting graphs to Doc2Vec documents...")
        self.graph_documents = self._create_doc2vec_documents(num_graphs)

        # Create training callback
        #training_callback = TrainingCallback()
        training_callback = TrainingCallback() # TODO: Decide on one, delete the other

        # Train Doc2Vec model
        print("Training Doc2Vec model...")
        self.doc2vec_model = Doc2Vec(
            documents=self.graph_documents,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            dm=self.dm,  # 0 = PV-DBOW (like original Graph2Vec)
            negative=self.negative_samples,
            sample=0.001,  # Subsampling threshold
            workers=self.num_workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            min_alpha=self.min_learning_rate,
            callbacks=[training_callback],
            compute_loss=True
        )

        # Evaluate model quality after training
        self.evaluate_model_quality()

        print(f"Training completed. Vocabulary size: {len(self.doc2vec_model.wv)}")
        return self

    # TODO: Check and remove
    def _create_doc2vec_documents_o(self, num_graphs: int) -> List[TaggedDocument]:
        """
        Convert cached subgraphs to Doc2Vec TaggedDocument format.

        Args:
            num_graphs: Number of graphs

        Returns:
            List of TaggedDocument objects for Doc2Vec training
        """
        documents = []

        for graph_id in tqdm(range(num_graphs), desc="Creating Doc2Vec documents"):
            if graph_id in self.cached_subgraphs:
                # Get subgraph strings for this graph
                subgraph_indices = self.cached_subgraphs[graph_id]

                # Convert indices back to subgraph strings
                subgraph_strings = []
                idx_to_subgraph = {idx: sg for sg, idx in self.subgraph_vocab.items()}

                for idx in subgraph_indices:
                    if idx in idx_to_subgraph:
                        subgraph_strings.append(idx_to_subgraph[idx])

                # Create TaggedDocument (words=subgraphs, tags=graph_id)
                if subgraph_strings:  # Only add if graph has subgraphs
                    doc = TaggedDocument(
                        words=subgraph_strings,
                        tags=[f"graph_{graph_id}"]
                    )
                    documents.append(doc)

        print(f"Created {len(documents)} documents for Doc2Vec training")
        return documents

    def _create_doc2vec_documents(self, num_graphs: int) -> List[TaggedDocument]:
        """
        Convert cached subgraphs to Doc2Vec TaggedDocument format.
        """
        documents = []

        # Pre-compute the reverse mapping once
        idx_to_subgraph = {idx: sg for sg, idx in self.subgraph_vocab.items()}

        for graph_id in tqdm(range(num_graphs), desc="Creating Doc2Vec documents"):
            if graph_id in self.cached_subgraphs:
                # Get subgraph strings for this graph
                subgraph_indices = self.cached_subgraphs[graph_id]

                # Convert indices to subgraph strings (much faster now)
                subgraph_strings = [idx_to_subgraph[idx] for idx in subgraph_indices
                                    if idx in idx_to_subgraph]

                # Create TaggedDocument (words=subgraphs, tags=graph_id)
                if subgraph_strings:  # Only add if graph has subgraphs
                    doc = TaggedDocument(
                        words=subgraph_strings,
                        tags=["graph_{}".format(graph_id)]
                    )
                    documents.append(doc)

        print("Created {} documents for Doc2Vec training".format(len(documents)))
        return documents

    def get_embeddings(self) -> torch.Tensor:
        """
        Get the learned graph embeddings from Doc2Vec model.

        Returns:
            Tensor of shape (num_graphs, embedding_dim) containing graph embeddings
        """
        if self.doc2vec_model is None:
            raise ValueError("Model has not been fitted yet")

        embeddings = []
        num_graphs = len(self.cached_subgraphs)

        for graph_id in range(num_graphs):
            tag = f"graph_{graph_id}"
            try:
                # Get document vector from Doc2Vec model
                embedding = self.doc2vec_model.dv[tag]
                embeddings.append(torch.tensor(embedding, dtype=torch.float32))
            except KeyError:
                # If graph wasn't in training (no subgraphs), return zero embedding
                print(f"Warning: Graph {graph_id} not found in model, using zero embedding")
                embeddings.append(torch.zeros(self.embedding_dim, dtype=torch.float32))

        return torch.stack(embeddings)

    def get_subgraph_embeddings(self) -> Optional[KeyedVectors]:
        """
        Get the learned subgraph (word) embeddings from Doc2Vec model.

        Returns:
            Gensim KeyedVectors object containing subgraph embeddings
        """
        if self.doc2vec_model is None:
            raise ValueError("Model has not been fitted yet")

        return self.doc2vec_model.wv

    def transform(self, dataset) -> torch.Tensor:
        """
        Transform new graphs to embeddings using the fitted Doc2Vec model.

        Args:
            dataset: PyTorch Geometric dataset or list of Data objects

        Returns:
            Tensor of embeddings for the input graphs
        """
        if self.doc2vec_model is None:
            raise ValueError("Model has not been fitted yet")

        # Create DataLoader for inference
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        all_embeddings = []

        for batch in tqdm(dataloader, desc="Transforming graphs"):
            batch_subgraphs = self._extract_subgraphs_batch(batch)

            # Process each graph in the batch
            for graph_subgraphs in batch_subgraphs:
                # Filter subgraphs that exist in vocabulary
                known_subgraphs = [sg for sg in graph_subgraphs if sg in self.subgraph_vocab]

                if known_subgraphs:
                    # Infer document vector for new graph
                    embedding = self.doc2vec_model.infer_vector(known_subgraphs)
                    all_embeddings.append(torch.tensor(embedding, dtype=torch.float32))
                else:
                    # If no known subgraphs, return zero embedding
                    all_embeddings.append(torch.zeros(self.embedding_dim, dtype=torch.float32))

        return torch.stack(all_embeddings)

    # Keep all your existing PyTorch Geometric helper methods unchanged
    def _extract_subgraphs_batch(self, batch: Batch) -> List[List[str]]:
        """Extract subgraphs using WL algorithm from PyTorch Geometric batch."""
        batch_subgraphs = []

        for i in range(len(batch.ptr) - 1):
            start_idx = batch.ptr[i].item()
            end_idx = batch.ptr[i + 1].item()
            num_nodes = end_idx - start_idx

            # Build adjacency list
            edge_mask = (batch.edge_index[0] >= start_idx) & (batch.edge_index[0] < end_idx)
            graph_edge_index = batch.edge_index[:, edge_mask] - start_idx

            adj_list = [[] for _ in range(num_nodes)]
            for j in range(graph_edge_index.size(1)):
                u, v = graph_edge_index[0, j].item(), graph_edge_index[1, j].item()
                adj_list[u].append(v)
                adj_list[v].append(u)

            # Get initial node labels
            if hasattr(batch, 'x') and batch.x is not None:
                current_features = {node: str(int(batch.x[start_idx + node, 0].item()))
                                    for node in range(num_nodes)}
            else:
                current_features = {node: str(len(adj_list[node]))
                                    for node in range(num_nodes)}
            # TODO: Maybe re-enable original implementation, but this is a test for graph2vec: adjacent nodes
            #current_features = {node: 1#str(len(adj_list[node]))
            #                    for node in range(num_nodes)}

            # Extract subgraphs using WL algorithm
            graph_subgraphs = self._get_wl_subgraphs(
                adj_list, current_features, num_nodes
            )

            batch_subgraphs.append(graph_subgraphs)

        return batch_subgraphs

    def _get_wl_subgraphs(self, adj_list: List[List[int]],
                          initial_features: Dict[int, str],
                          num_nodes: int) -> List[str]:
        """WL implementation matching the original NetworkX version."""
        all_subgraphs = []
        current_features = initial_features.copy()

        # Add degree 0 subgraphs (just node labels)
        for node in range(num_nodes):
            all_subgraphs.append(current_features[node])

        # Iterative WL relabeling
        for iteration in range(self.max_degree):
            new_features = {}

            for node in range(num_nodes):
                # Get neighbor features
                neighbor_features = [current_features[neighbor] for neighbor in adj_list[node]]

                # Create feature string (matching original format)
                features = [str(current_features[node])] + sorted([str(feat) for feat in neighbor_features])
                features_str = "_".join(features)

                # Use MD5 hash like original (deterministic)
                hash_object = hashlib.md5(features_str.encode())
                new_features[node] = hash_object.hexdigest()

            # Add all subgraphs from this iteration
            for node in range(num_nodes):
                all_subgraphs.append(new_features[node])

            current_features = new_features

        return all_subgraphs

    def _cache_all_subgraphs(self, dataset, num_graphs: int) -> Dict[int, List[int]]:
        """Pre-extract and cache subgraphs for all graphs."""
        cached_subgraphs = {}

        # Create DataLoader for caching
        cache_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        graph_idx = 0
        for batch in tqdm(cache_dataloader, desc="Caching subgraphs"):
            batch_subgraphs = self._extract_subgraphs_batch(batch)

            # Store subgraphs for each graph in the batch
            for i, graph_subgraphs in enumerate(batch_subgraphs):
                if graph_idx < num_graphs:
                    # Convert subgraph strings to indices
                    subgraph_indices = [self.subgraph_vocab[sg] for sg in graph_subgraphs
                                        if sg in self.subgraph_vocab]
                    cached_subgraphs[graph_idx] = subgraph_indices
                    graph_idx += 1

        return cached_subgraphs

    def _build_vocabulary(self, dataloader: DataLoader) -> Dict[str, int]:
        """Build vocabulary with proper contiguous indexing."""
        # Try to load cached vocabulary
        if self.vocab_cache_path and os.path.exists(self.vocab_cache_path):
            print(f"Loading cached vocabulary from {self.vocab_cache_path}")
            with open(self.vocab_cache_path, 'rb') as f:
                return pickle.load(f)

        print("Building vocabulary from dataset...")
        subgraph_counter = Counter()

        for batch in tqdm(dataloader, desc="Processing batches for vocabulary"):
            batch_subgraphs = self._extract_subgraphs_batch(batch)

            # Count subgraphs
            for graph_subgraphs in batch_subgraphs:
                for subgraph in graph_subgraphs:
                    subgraph_counter[subgraph] += 1

        # Create vocabulary with proper contiguous indexing
        filtered_subgraphs = [(sg, count) for sg, count in subgraph_counter.items()
                              if count >= self.min_count]
        vocab = {sg: idx for idx, (sg, count) in enumerate(filtered_subgraphs)}

        # Cache vocabulary if path provided
        if self.vocab_cache_path:
            print(f"Caching vocabulary to {self.vocab_cache_path}")
            os.makedirs(os.path.dirname(self.vocab_cache_path), exist_ok=True)
            with open(self.vocab_cache_path, 'wb') as f:
                pickle.dump(vocab, f)

        print(f"Created vocabulary with {len(vocab)} subgraphs")
        return vocab

    def save_model(self, path: str):
        """Save the trained model."""
        if self.doc2vec_model is None:
            raise ValueError("Model has not been fitted yet")

        # Save Doc2Vec model
        self.doc2vec_model.save(f"{path}_doc2vec.model")

        # Save additional metadata
        metadata = {
            'subgraph_vocab': self.subgraph_vocab,
            'vocab_size': self.vocab_size,
            'cached_subgraphs': self.cached_subgraphs,
            'config': {
                'embedding_dim': self.embedding_dim,
                'max_degree': self.max_degree,
                'negative_samples': self.negative_samples,
                'min_count': self.min_count,
                'window': self.window,
                'dm': self.dm
            }
        }

        with open(f"{path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

    def load_model(self, path: str):
        """Load a trained model."""
        # Load Doc2Vec model
        self.doc2vec_model = Doc2Vec.load(f"{path}_doc2vec.model")

        # Load metadata
        with open(f"{path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        self.subgraph_vocab = metadata['subgraph_vocab']
        self.vocab_size = metadata['vocab_size']
        self.cached_subgraphs = metadata['cached_subgraphs']

        # Update config
        config = metadata['config']
        self.embedding_dim = config['embedding_dim']
        self.max_degree = config['max_degree']
        self.negative_samples = config['negative_samples']
        self.min_count = config['min_count']
        self.window = config['window']
        self.dm = config['dm']

    # TODO: Redo evaluation
    def evaluate_model_quality(self, test_graphs=None, num_test_graphs=100):
        """
        Comprehensive evaluation of the trained Graph2Vec model.
        """
        if self.doc2vec_model is None:
            raise ValueError("Model has not been fitted yet")

        print("=" * 60)
        print("GRAPH2VEC MODEL QUALITY ASSESSMENT")
        print("=" * 60)

        # 1. Vocabulary Statistics
        self._evaluate_vocabulary_quality()

        # 2. Embedding Quality
        self._evaluate_embedding_quality()

        # 3. Similarity Analysis
        self._evaluate_similarity_patterns()

        # 4. Convergence Analysis
        if hasattr(self, 'training_callback') and hasattr(self.training_callback,
                                                          'losses') and self.training_callback.losses:
            self._evaluate_convergence()

        # 5. Reconstruction Quality (if test graphs provided)
        if test_graphs is not None:
            self._evaluate_reconstruction_quality(test_graphs, num_test_graphs)

    def _evaluate_vocabulary_quality(self):
        """Evaluate the quality of the subgraph vocabulary."""
        print("\n1. VOCABULARY QUALITY:")
        print("-" * 30)

        vocab_size = len(self.subgraph_vocab)
        doc2vec_vocab_size = len(self.doc2vec_model.wv)

        print(f"Subgraph vocabulary size: {vocab_size}")
        print(f"Doc2Vec vocabulary size: {doc2vec_vocab_size}")
        print(f"Vocabulary utilization: {doc2vec_vocab_size / vocab_size * 100:.1f}%")

        # Analyze subgraph frequency distribution
        subgraph_counts = Counter()
        for doc in self.graph_documents:
            for word in doc.words:
                subgraph_counts[word] += 1

        if subgraph_counts:
            frequencies = list(subgraph_counts.values())
            print(f"Most frequent subgraph appears {max(frequencies)} times")
            print(f"Least frequent subgraph appears {min(frequencies)} times")
            print(f"Average subgraph frequency: {np.mean(frequencies):.2f}")
            print(f"Subgraph frequency std: {np.std(frequencies):.2f}")

    def _evaluate_embedding_quality(self):
        """Evaluate the quality of learned embeddings."""
        print("\n2. EMBEDDING QUALITY:")
        print("-" * 30)

        # Get all graph embeddings
        embeddings = self.get_embeddings().numpy()

        # Basic statistics
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Number of graph embeddings: {embeddings.shape[0]}")
        print(f"Embedding mean: {np.mean(embeddings):.6f}")
        print(f"Embedding std: {np.std(embeddings):.6f}")
        print(f"Embedding range: [{np.min(embeddings):.6f}, {np.max(embeddings):.6f}]")

        # Check for degenerate embeddings (all zeros or all same)
        zero_embeddings = np.sum(np.all(embeddings == 0, axis=1))
        if zero_embeddings > 0:
            print(f"WARNING: {zero_embeddings} graphs have zero embeddings!")

        # Check embedding diversity
        pairwise_similarities = []
        sample_size = min(100, len(embeddings))
        indices = np.random.choice(len(embeddings), sample_size, replace=False)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                sim = np.dot(embeddings[indices[i]], embeddings[indices[j]]) / (
                        np.linalg.norm(embeddings[indices[i]]) * np.linalg.norm(embeddings[indices[j]]) + 1e-8
                )
                pairwise_similarities.append(sim)

        if pairwise_similarities:
            print(f"Average pairwise cosine similarity: {np.mean(pairwise_similarities):.6f}")
            print(f"Similarity std: {np.std(pairwise_similarities):.6f}")

            if np.mean(pairwise_similarities) > 0.9:
                print("WARNING: Very high average similarity - embeddings may be too similar!")
            elif np.mean(pairwise_similarities) < 0.1:
                print("GOOD: Low average similarity indicates diverse embeddings")

    def _evaluate_similarity_patterns(self):
        """Evaluate similarity patterns in the embedding space."""
        print("\n3. SIMILARITY PATTERNS:")
        print("-" * 30)

        # Test similarity queries
        num_graphs = len(self.cached_subgraphs)
        if num_graphs < 10:
            print("Not enough graphs for similarity analysis")
            return

        # Sample some graphs and find their most similar neighbors
        sample_graphs = np.random.choice(num_graphs, min(5, num_graphs), replace=False)

        for graph_id in sample_graphs:
            try:
                similar_graphs = self.most_similar_graphs(graph_id, topn=3)
                print(f"Graph {graph_id} most similar to: {similar_graphs}")
            except Exception as e:
                print(f"Could not compute similarities for graph {graph_id}: {str(e)}")

    def _evaluate_convergence(self):
        """Evaluate training convergence."""
        print("\n4. CONVERGENCE ANALYSIS:")
        print("-" * 30)

        losses = self.training_callback.losses
        if len(losses) < 2:
            print("Insufficient loss data for convergence analysis")
            return

        # Check if loss is decreasing
        loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]
        print(f"Loss trend (slope): {loss_trend:.6f}")

        if loss_trend < 0:
            print("GOOD: Loss is decreasing over time")
        else:
            print("WARNING: Loss is not decreasing - model may not be converging")

        # Check for convergence in last few epochs
        if len(losses) >= 5:
            recent_losses = losses[-5:]
            recent_std = np.std(recent_losses)
            print(f"Recent loss stability (std of last 5): {recent_std:.6f}")

            if recent_std < 0.01:
                print("GOOD: Loss has stabilized")
            else:
                print("INFO: Loss still changing - may need more epochs")

    def _evaluate_reconstruction_quality(self, test_graphs, num_test_graphs):
        """Evaluate how well the model can reconstruct/represent test graphs."""
        print("\n5. RECONSTRUCTION QUALITY:")
        print("-" * 30)

        # Transform test graphs and check embedding quality
        try:
            test_embeddings = self.transform(test_graphs[:num_test_graphs])

            # Check for zero embeddings (indicates poor vocabulary coverage)
            zero_test_embeddings = torch.sum(torch.all(test_embeddings == 0, dim=1)).item()
            coverage = (num_test_graphs - zero_test_embeddings) / num_test_graphs * 100

            print(f"Test graph coverage: {coverage:.1f}%")
            print(f"Graphs with zero embeddings: {zero_test_embeddings}/{num_test_graphs}")

            if coverage > 80:
                print("GOOD: High vocabulary coverage on test graphs")
            elif coverage > 50:
                print("MODERATE: Reasonable vocabulary coverage")
            else:
                print("WARNING: Low vocabulary coverage - may need larger vocabulary")

        except Exception as e:
            print(f"Could not evaluate reconstruction quality: {str(e)}")

    def similarity(self, graph_id1: int, graph_id2: int) -> float:
        """Calculate similarity between two graphs."""
        if self.doc2vec_model is None:
            raise ValueError("Model has not been fitted yet")

        tag1 = f"graph_{graph_id1}"
        tag2 = f"graph_{graph_id2}"

        try:
            vec1 = self.doc2vec_model.dv[tag1]
            vec2 = self.doc2vec_model.dv[tag2]

            # Cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
        except KeyError as e:
            print(f"Graph not found: {e}")
            return 0.0

    def most_similar_graphs(self, graph_id: int, topn: int = 5) -> List[Tuple[str, float]]:
        """Find most similar graphs to the given graph."""
        if self.doc2vec_model is None:
            raise ValueError("Model has not been fitted yet")

        tag = f"graph_{graph_id}"

        try:
            # Get similar document vectors
            similar = self.doc2vec_model.dv.most_similar(tag, topn=topn)
            return similar
        except KeyError:
            print(f"Graph {graph_id} not found in model")
            return []
