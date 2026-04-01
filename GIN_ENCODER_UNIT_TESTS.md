# GIN Encoder Unit Test Implementation Guide

## Context

This repository contains a VAMP (Variational Approach to Markov Processes) pipeline that uses a GIN (Graph Isomorphism Network) as its encoder. We need a comprehensive unit test suite to validate the GIN encoder is implemented correctly, both in isolation and as part of the VAMP pipeline.

Use `pytest` as the test framework. Use PyTorch Geometric datasets and utilities — they are already a project dependency.

---

## Test File Structure

Create a test file (e.g., `tests/test_gin_encoder.py`) that imports the project's actual GIN encoder. Discover where the GIN encoder class is defined in this repository, how it is instantiated, and what its forward signature looks like (likely `forward(x, edge_index, batch=None)` or similar). Adapt all tests to match the real interface.

---

## Required Test Categories

### 1. Basic Sanity

- **Single graph forward pass**: Pass one graph from `TUDataset("MUTAG")` through the encoder. Assert the output shape is `(1, output_dim)` and dtype is `float32`.
- **Batched forward pass**: Batch 8 MUTAG graphs using `torch_geometric.data.Batch.from_data_list`. Assert output shape is `(8, output_dim)`.
- **No NaN/Inf**: Run 50 MUTAG graphs through and assert `torch.isfinite(out).all()`.
- **Non-degenerate output**: Run 20 different graphs. Assert that `torch.unique(out, dim=0)` has more than 1 row — the encoder must not collapse everything to the same embedding.

### 2. Permutation Invariance

This is the most important structural test. GIN with a sum/mean readout must be permutation invariant at the graph level.

For several graphs from both MUTAG and PROTEINS datasets:
1. Record the original graph-level embedding.
2. Create a permuted copy: randomly shuffle node indices, remap `edge_index` accordingly, reorder `x`.
3. Assert the graph-level embeddings are equal within tolerance (`atol=1e-5`).

Implementation hint for permuting a `Data` object:
```python
perm = torch.randperm(data.x.size(0))
new_x = data.x[perm]
inv_perm = torch.empty_like(perm)
inv_perm[perm] = torch.arange(data.x.size(0))
new_edge_index = inv_perm[data.edge_index]
```

### 3. Gradient Flow

- **All parameters receive gradients**: Do a forward pass, call `.sum().backward()` on the output, then iterate over `model.named_parameters()`. Assert no parameter has `grad is None` or `(grad == 0).all()`.
- **Gradient magnitudes are reasonable**: On a batch of 16 graphs, do a forward + backward. Assert no gradient norm exceeds `1e6` (explosion). For non-batchnorm weight parameters, assert gradient norm exceeds `1e-12` (vanishing).

### 4. WL Expressiveness

GIN should be as expressive as the 1-WL isomorphism test. Test this with known graph pairs.

- **C6 vs 2×C3**: Construct a 6-cycle and two disjoint triangles. Both have 6 nodes and are 3-regular, but they are non-isomorphic and 1-WL distinguishable. Use constant `ones` node features. Try up to 10 random weight initializations — at least one should produce different embeddings for the two graphs. If none do, the architecture may lack WL-level expressiveness.
- **Different-sized cycles**: A 5-cycle and a 7-cycle (same constant features) must always produce different embeddings due to the sum aggregation over different node counts.

### 5. Determinism

Set the model to `eval()` mode. Run the same graph through twice. Assert the outputs are bitwise identical (`atol=0, rtol=0`). This catches accidental stochastic behavior (e.g., dropout left on in eval).

### 6. Learnability

These tests verify the encoder can actually learn, not just produce outputs.

- **Loss decreases on MUTAG**: Attach a simple linear classification head on top of the encoder (mapping to `dataset.num_classes`). Train for 20 epochs on 100 MUTAG graphs with `Adam(lr=1e-3)` and `CrossEntropyLoss`. Assert the final loss is less than 80% of the initial loss.
- **Overfitting a tiny batch**: Take 5 MUTAG graphs. Train for 200 steps. Assert classification accuracy on those same 5 graphs is at least 80%. If the encoder cannot memorize 5 graphs, something is broken.

### 7. Cross-Dataset Shape Compatibility

Load `TUDataset("PROTEINS")` (3 node features, different from MUTAG's 7). Instantiate the encoder with `in_channels=3`. Run a batch of 10 PROTEINS graphs. Assert correct output shape and no NaN/Inf. This catches any hard-coded feature dimensions.

### 8. Edge Cases

Build synthetic `Data` objects for these degenerate cases. Each should produce a finite output of the correct shape without raising exceptions.

- **Single node, no edges**: `x` is shape `(1, in_channels)`, `edge_index` is shape `(2, 0)`.
- **Disconnected graph**: 4 nodes, only nodes 0 and 1 are connected. Nodes 2 and 3 are isolated.
- **Self-loops**: Include `(i, i)` edges alongside normal edges.
- **Large graph stress test**: 1000 nodes, ~5000 random edges. Ensure no crash or memory issue and output is finite.

---

## Implementation Notes

- Use `@pytest.fixture(scope="module")` for dataset loading (MUTAG, PROTEINS) to avoid re-downloading per test.
- Use a factory fixture for the encoder so it is easy to swap between different configurations.
- Always call `model.eval()` and use `torch.no_grad()` for inference tests. Only use `model.train()` for gradient and learnability tests.
- TUDataset downloads automatically to the path you specify (e.g., `/tmp/TUDataset`). This is fine for CI.
- If the project's GIN encoder has a different readout than sum pooling (e.g., mean pooling), adjust the permutation invariance and WL expressiveness expectations accordingly. Mean pooling is still permutation invariant but may not distinguish C6 from 2×C3 — note this in a comment if relevant.

---

## Optional: VAMP Integration Smoke Tests

If feasible, also add a lightweight integration test that runs the GIN encoder within the VAMP loss computation:

- Construct a minimal pair of batches (time `t` and time `t + lag`) from synthetic or toy graph data.
- Compute the VAMP-2 score using the encoder's output.
- Assert the score is a finite positive scalar.
- Do one backward pass and verify gradients flow through the encoder.

This does not validate correctness of the VAMP score itself, but it ensures the encoder is properly wired into the pipeline.

---

## Datasets Used

| Dataset  | Graphs | Node Features | Use Case                             |
|----------|--------|---------------|--------------------------------------|
| MUTAG    | 188    | 7             | Primary test dataset, small and fast |
| PROTEINS | 1113   | 3             | Cross-dataset shape compatibility    |

Both are loaded via `torch_geometric.datasets.TUDataset`.

---

## Success Criteria

All tests pass. Pay special attention to:
- Permutation invariance tests (category 2) — failure here means the encoder or readout is fundamentally broken.
- Gradient flow tests (category 3) — failure means silent dead layers.
- WL expressiveness (category 4) — failure suggests the GIN aggregation or MLP structure deviates from the paper.
