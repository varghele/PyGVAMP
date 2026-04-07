# PyGVAMP Testing Guide

This document outlines unit testing best practices for PyGVAMP and explains the test structure.

## What Makes a Good Unit Test

### 1. Isolation

Each test runs independently without depending on other tests or shared state. Use pytest fixtures to create fresh test data for each test.

```python
@pytest.fixture
def single_graph_data(device, seed):
    """Create fresh test data for each test."""
    num_nodes = 10
    x = torch.randn(num_nodes, 20, device=device)
    # ... create graph data
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
```

### 2. Fast Execution

Unit tests should run quickly (milliseconds to a few seconds) to enable frequent testing during development. The 26 encoder tests run in ~2 seconds.

**Tips:**
- Use small graph sizes (10-20 nodes)
- Use CPU for deterministic tests
- Minimize model complexity (2 layers, small hidden dims)

### 3. Deterministic Results

Use fixed random seeds to ensure reproducible results:

```python
@pytest.fixture
def seed():
    """Fixed seed for reproducibility."""
    torch.manual_seed(42)
    return 42
```

### 4. Clear Naming

Test names should describe what's being tested and expected behavior:

| Name | Meaning |
|------|---------|
| `test_forward_single_graph` | Tests forward pass on a single graph |
| `test_gradient_flow` | Tests that gradients propagate to parameters |
| `test_gradient_flow_to_inputs` | Tests gradients flow back to input tensors |
| `test_embedding_types` | Tests all embedding type options work |

### 5. Single Responsibility

Each test verifies one specific behavior. Don't combine multiple unrelated assertions:

```python
# Good: One behavior per test
def test_forward_produces_correct_shape(self, data, device):
    output, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
    assert output.shape == (1, 32)

def test_forward_produces_no_nan(self, data, device):
    output, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
    assert not torch.isnan(output).any()
```

### 6. Arrange-Act-Assert (AAA) Pattern

Structure tests in three clear phases:

```python
def test_gradient_flow_to_inputs(self, batched_graph_data, device):
    # ARRANGE: Set up model and data
    batch, num_graphs = batched_graph_data
    model = SchNetEncoderNoEmbed(
        node_dim=batch.x.size(1),
        edge_dim=batch.edge_attr.size(1),
        hidden_dim=64,
        output_dim=32,
        n_interactions=2
    ).to(device)

    # ACT: Execute the operation
    output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    loss = output.sum()
    loss.backward()

    # ASSERT: Verify expected behavior
    assert batch.x.grad is not None, "No gradient for node features"
    assert not torch.all(batch.x.grad == 0), "Zero gradient for node features"
```

### 7. Test Edge Cases

Include tests for boundary conditions and unusual inputs:

- **Single node graphs** - Minimum valid input
- **Missing batch tensors** - Optional parameter handling
- **Large batches** - Scalability verification
- **Empty inputs** - Error handling (if applicable)

### 8. Informative Failure Messages

When assertions fail, provide context about what went wrong:

```python
# Good: Informative message
assert output.shape == (num_graphs, 32), f"Expected ({num_graphs}, 32), got {output.shape}"

# Bad: No context
assert output.shape == (num_graphs, 32)
```

### 9. Document Expected Behaviors

Use docstrings and comments to explain what each test verifies:

```python
def test_gradient_flow(self, batched_graph_data, device):
    """
    Test that gradients flow through all parameters.

    Note:
        Some parameters (e.g., global_mlp in layer 0) may not receive
        gradients due to architectural choices. This is expected.
    """
```

## Test Structure

```
tests/
├── __init__.py              # Package marker
├── README.md                # Test summary and run instructions
├── test_encoders.py         # Encoder unit tests
├── test_dataset.py          # Dataset creation tests (TODO)
├── test_vamp_score.py       # VAMP score calculation tests (TODO)
└── test_pipeline.py         # Integration tests (TODO)
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific file
pytest tests/test_encoders.py -v

# Run specific class
pytest tests/test_encoders.py::TestSchNetEncoder -v

# Run specific test
pytest tests/test_encoders.py::TestSchNetEncoder::test_forward_single_graph -v
```

### Useful Options

```bash
# Short traceback on failures
pytest tests/ -v --tb=short

# Stop on first failure
pytest tests/ -v -x

# Run last failed tests
pytest tests/ -v --lf

# Show print statements
pytest tests/ -v -s

# Run tests matching pattern
pytest tests/ -v -k "gradient"
```

### Coverage (requires pytest-cov)

```bash
# Install coverage plugin
pip install pytest-cov

# Run with coverage report
pytest tests/ --cov=pygv --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=pygv --cov-report=html
```

## Writing New Tests

### Template for Encoder Tests

```python
class TestMyEncoder:
    """Tests for MyEncoder."""

    def test_forward_single_graph(self, single_graph_data, device):
        """Test forward pass on a single graph."""
        data = single_graph_data

        model = MyEncoder(
            node_dim=data.x.size(1),
            edge_dim=data.edge_attr.size(1),
            # ... other params
        ).to(device)

        # Use eval mode for single graph (BatchNorm compatibility)
        model.eval()
        with torch.no_grad():
            output, aux = model(data.x, data.edge_index, data.edge_attr, data.batch)

        assert output.shape == (1, expected_dim)
        assert not torch.isnan(output).any()

    def test_forward_batched(self, batched_graph_data, device):
        """Test forward pass on batched graphs."""
        batch, num_graphs = batched_graph_data

        model = MyEncoder(...).to(device)
        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        assert output.shape == (num_graphs, expected_dim)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, batched_graph_data, device):
        """Test that gradients flow through all parameters."""
        batch, num_graphs = batched_graph_data

        model = MyEncoder(...).to(device)
        output, aux = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = output.sum()

        assert_gradients_flow(model, loss, "MyEncoder")
```

### Handling BatchNorm with Single Graphs

BatchNorm requires batch_size > 1 in training mode. For single graph tests:

```python
def test_forward_single_graph(self, single_graph_data, device):
    model = MyEncoder(...).to(device)

    # Switch to eval mode to avoid BatchNorm issues
    model.eval()
    with torch.no_grad():
        output, aux = model(...)
```

For gradient tests, use batched data instead of single graphs.

## Continuous Integration

To set up CI/CD:

1. Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ -v --tb=short
```

---

*Last updated: 2026-01-16*
