import torch
from torch_geometric.utils import softmax

# Add to your comparison script
def test_softmax_implementations():
    # Create test data
    torch.manual_seed(42)
    n_msgs = 1000
    msg_dim = 32

    # Sample messages and target indices
    messages = torch.randn(n_msgs, msg_dim)
    targets = torch.randint(0, 100, (n_msgs,))  # Assign to 100 different targets
    edge_index_i = targets  # For PyG's softmax

    # Compute attention scores
    attention_vector = torch.randn(msg_dim, 1)
    scores = torch.matmul(messages, attention_vector).squeeze(-1)

    # Method 1: PyG's softmax
    norm_attn_pyg = softmax(scores, edge_index_i)

    # Method 2: Manual softmax
    norm_attn_manual = torch.zeros_like(scores)
    for node in torch.unique(targets):
        mask = (targets == node)
        node_scores = scores[mask]
        node_attn = torch.softmax(node_scores, dim=0)
        norm_attn_manual[mask] = node_attn

    # Compare
    diff = (norm_attn_pyg - norm_attn_manual).abs()
    print(f"Softmax difference - Mean: {diff.mean().item():.8f}, Max: {diff.max().item():.8f}")


def test_aggregation_difference():
    # Create test data
    torch.manual_seed(42)
    n_msgs = 10000
    msg_dim = 32
    n_nodes = 1000

    # Generate messages and target indices
    messages = torch.randn(n_msgs, msg_dim)
    target_indices = torch.randint(0, n_nodes, (n_msgs,))

    # Method 1: PyG's scatter_add
    output_pyg = torch.zeros(n_nodes, msg_dim)
    from torch_scatter import scatter_add
    output_pyg = scatter_add(messages, target_indices, dim=0, dim_size=n_nodes)

    # Method 2: Manual loop aggregation
    output_manual = torch.zeros(n_nodes, msg_dim)
    for i in range(len(messages)):
        output_manual[target_indices[i]] += messages[i]

    # Compare
    diff = (output_pyg - output_manual).abs()
    print(f"Aggregation difference - Mean: {diff.mean().item():.8f}, Max: {diff.max().item():.8f}")



test_softmax_implementations()
test_aggregation_difference()