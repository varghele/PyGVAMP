import torch
import numpy as np


# Simple LinearLayer implementation for testing
class LinearLayer:
    @staticmethod
    def create(d_in, d_out, bias=True, activation=None):
        layers = [torch.nn.Linear(d_in, d_out, bias=bias)]
        if activation is not None:
            layers.append(activation)
        return layers


# Simple ContinuousFilterConv implementation for testing
class ContinuousFilterConv(torch.nn.Module):
    def __init__(self, n_gaussians, n_filters, activation, normalization_layer=None):
        super(ContinuousFilterConv, self).__init__()
        self.filter_generator = torch.nn.Sequential(
            torch.nn.Linear(n_gaussians, n_filters),
            activation,
            torch.nn.Linear(n_filters, n_filters)
        )
        self.normalization_layer = normalization_layer
        self.use_attention = True
        if self.use_attention:
            self.nbr_filter = torch.nn.Parameter(torch.Tensor(n_filters, 1))
            torch.nn.init.xavier_uniform_(self.nbr_filter)

    def forward(self, features, rbf_expansion, neighbor_list):
        # Simplified implementation for testing
        batch_size, n_atoms, _ = features.shape
        n_neighbors = neighbor_list.shape[2]

        # Generate filters
        filters = self.filter_generator(rbf_expansion)

        # Gather neighbor features
        neighbor_list_expanded = neighbor_list.reshape(batch_size, n_atoms * n_neighbors, 1)
        neighbor_list_expanded = neighbor_list_expanded.expand(-1, -1, features.shape[2])
        neighbor_features = torch.gather(features, 1, neighbor_list_expanded)
        neighbor_features = neighbor_features.reshape(batch_size, n_atoms, n_neighbors, -1)

        # Apply filters
        filtered_features = neighbor_features * filters

        # Apply attention if used
        if self.use_attention:
            attention = torch.matmul(filtered_features, self.nbr_filter).squeeze(-1)
            attention = torch.softmax(attention, dim=-1)
            output = torch.einsum('bij,bijc->bic', attention, filtered_features)
            return output, attention
        else:
            # Sum over neighbors
            output = filtered_features.sum(dim=2)
            return output, None


# Classic InteractionBlock implementation
class ClassicInteractionBlock(torch.nn.Module):
    """SchNet interaction block as described by Schütt et al. (2018)."""

    def __init__(self, n_inputs, n_gaussians, n_filters, activation=torch.nn.Tanh()):
        super(ClassicInteractionBlock, self).__init__()

        # Initial dense layer
        self.initial_dense = torch.nn.Sequential(
            *LinearLayer.create(n_inputs, n_filters, bias=False, activation=None)
        )

        # Continuous filter convolution
        self.cfconv = ContinuousFilterConv(
            n_gaussians=n_gaussians,
            n_filters=n_filters,
            activation=activation
        )

        # Output layers
        output_layers = []
        output_layers.extend(LinearLayer.create(n_filters, n_filters, bias=True, activation=activation))
        output_layers.extend(LinearLayer.create(n_filters, n_filters, bias=True))
        self.output_dense = torch.nn.Sequential(*output_layers)

    def forward(self, features, rbf_expansion, neighbor_list):
        init_feature_output = self.initial_dense(features)
        conv_output, attn = self.cfconv(init_feature_output.to(torch.float32),
                                        rbf_expansion.to(torch.float32),
                                        neighbor_list)
        output_features = self.output_dense(conv_output).to(torch.float32)
        return output_features, attn


# Create minimal MLP implementation
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, act='tanh'):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()

        # Create layers
        if num_layers == 1:
            # Single layer
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            # First layer
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))

            # Get activation function
            if isinstance(act, str):
                if act == 'tanh':
                    activation = torch.nn.Tanh()
                else:
                    activation = torch.nn.ReLU()
            else:
                activation = act

            # Add activation after first layer
            self.acts.append(activation)

            # Middle layers
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
                self.acts.append(activation)

            # Final layer
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.acts):
                x = self.acts[i](x)
        return x


# PyG CFConv implementation
class CFConv(torch.nn.Module):
    """Continuous Filter Convolution for PyTorch Geometric."""

    def __init__(self, in_channels, out_channels, edge_channels, hidden_channels, activation='tanh',
                 use_attention=True):
        super(CFConv, self).__init__()

        # Set channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convert string activation to module if needed
        if isinstance(activation, str):
            if activation == 'tanh':
                self.activation = torch.nn.Tanh()
            else:
                self.activation = torch.nn.ReLU()
        else:
            self.activation = activation

        # Filter network transforms edge features to weights
        self.filter_network = MLP(
            in_channels=edge_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=2,
            act=self.activation
        )

        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention_vector = torch.nn.Parameter(torch.Tensor(out_channels, 1))
            torch.nn.init.xavier_uniform_(self.attention_vector, gain=1.414)

    def forward(self, x, edge_index, edge_attr):
        """Forward pass with calculations similar to classic CFConv."""
        # Generate weights from edge attributes
        edge_weights = self.filter_network(edge_attr)

        # Extract source and target nodes
        source = edge_index[0]
        target = edge_index[1]

        # Get feature vectors for source nodes
        source_features = x[source]

        # Compute messages as product of source features and edge weights
        messages = source_features * edge_weights

        # Initialize attention weights
        attention_weights = None

        # Apply attention if enabled
        if self.use_attention:
            # Compute attention scores
            attn_scores = torch.matmul(messages, self.attention_vector).squeeze(-1)

            # Group by target node for softmax
            target_nodes = torch.unique(target)
            attention_weights = torch.zeros_like(attn_scores)

            # Apply softmax per target node
            for node in target_nodes:
                mask = (target == node)
                node_scores = attn_scores[mask]
                node_attention = torch.softmax(node_scores, dim=0)
                attention_weights[mask] = node_attention

            # Apply attention to messages
            messages = messages * attention_weights.unsqueeze(-1)

        # Sum messages for each target node
        output = torch.zeros((x.size(0), self.out_channels), device=x.device)
        for i in range(len(source)):
            output[target[i]] += messages[i]

        return output, attention_weights


# GCNInteraction for PyG
class GCNInteraction(torch.nn.Module):
    """SchNet-style interaction block using PyTorch Geometric CFConv."""

    def __init__(self, in_channels, edge_channels, hidden_channels, activation='tanh', use_attention=True):
        super(GCNInteraction, self).__init__()

        # Initial dense layer (matching classic implementation)
        self.initial_dense = torch.nn.Linear(in_channels, in_channels, bias=False)

        # CFConv layer - use in_channels for both input and output to maintain dimensions
        self.cfconv = CFConv(
            in_channels=in_channels,
            out_channels=in_channels,  # Same as input to preserve dimensions
            edge_channels=edge_channels,
            hidden_channels=hidden_channels,
            activation=activation,
            use_attention=use_attention
        )

        # Output transformation
        self.output_layer = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels,  # Back to input dimension for residual
            num_layers=2,
            act=activation
        )

    def forward(self, x, edge_index, edge_attr):
        """Forward pass for interaction block."""
        # Initial dense layer (matching classic implementation)
        h = self.initial_dense(x)

        # Apply continuous filter convolution
        conv_output, attention = self.cfconv(h, edge_index, edge_attr)

        # Transform back to input dimension
        output = self.output_layer(conv_output)

        return output, attention


# SchNetEncoderNoEmbed implementation
class SchNetEncoderNoEmbed(torch.nn.Module):
    """SchNet encoder using PyTorch Geometric native operations."""

    def __init__(
            self,
            node_dim,
            edge_dim,
            hidden_dim=64,
            output_dim=32,
            n_interactions=3,
            activation='tanh',
            use_attention=True
    ):
        super(SchNetEncoderNoEmbed, self).__init__()

        self.node_dim = node_dim
        self.output_dim = output_dim
        self.use_residual = True  # Add flag to control residual connections

        # Interaction blocks
        self.interactions = torch.nn.ModuleList([
            GCNInteraction(
                in_channels=node_dim,
                edge_channels=edge_dim,
                hidden_channels=hidden_dim,
                activation=activation,
                use_attention=use_attention
            ) for _ in range(n_interactions)
        ])

        # Output network for graph-level readout
        self.output_network = MLP(
            in_channels=node_dim,  # Use node_dim as each interaction preserves dimensions
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=2,
            act=activation
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Forward pass for SchNet encoder."""
        # If no batch information provided, assume a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Node features start as x (no initial embedding)
        h = x

        # Store attention weights from each layer
        attentions = []

        # Store deltas for debugging
        deltas = []

        # Apply interaction blocks with residual connections
        for interaction in self.interactions:
            delta, attention = interaction(h, edge_index, edge_attr)
            deltas.append(delta)  # Store delta for debugging

            if self.use_residual:
                h = h + delta  # Residual connection
            else:
                h = delta  # Direct assignment

            if attention is not None:
                attentions.append(attention)

        # Global pooling (mean of node features per graph)
        pooled = torch.zeros((int(batch.max().item()) + 1, h.size(1)), device=h.device)
        for i in range(h.size(0)):
            pooled[batch[i]] += h[i]
        node_counts = torch.bincount(batch)
        pooled = pooled / node_counts.unsqueeze(1).float()

        # Final output transformation
        output = self.output_network(pooled)

        return output, (h, attentions, deltas)


def run_debug_comparison():
    """Run a debug comparison between classic and PyG implementations."""
    print("\n===== SchNet Implementation Debug Comparison =====")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parameters for test
    batch_size = 2
    n_atoms = 20
    n_neighbors = 10
    n_features = 32
    n_gaussians = 16

    print(f"Generating test data with {batch_size} batches, {n_atoms} atoms, {n_features} features")

    # Create data for classic implementation
    classic_features = torch.randn(batch_size, n_atoms, n_features)
    classic_rbf = torch.randn(batch_size, n_atoms, n_neighbors, n_gaussians)

    # Create neighbor list - each atom has n_neighbors
    classic_neighbors = torch.zeros(batch_size, n_atoms, n_neighbors, dtype=torch.long)

    # For each atom, select n_neighbors other atoms as neighbors
    for b in range(batch_size):
        for i in range(n_atoms):
            for j in range(n_neighbors):
                # Choose neighbors in a cyclic pattern
                classic_neighbors[b, i, j] = (i + j + 1) % n_atoms

    # Create PyG compatible data
    pyg_nodes = []
    edge_src = []
    edge_dst = []
    edge_attr = []
    batch_indices = []

    for b in range(batch_size):
        # Add nodes for this batch
        for i in range(n_atoms):
            # Node features
            pyg_nodes.append(classic_features[b, i])
            batch_indices.append(b)  # Track batch index for each node

            # Create edges for each neighbor in classical neighbor list
            for j in range(n_neighbors):
                neighbor_idx = classic_neighbors[b, i, j].item()
                # Source is neighbor, destination is current node (matching message passing convention)
                src = neighbor_idx + b * n_atoms  # Offset by batch
                dst = i + b * n_atoms  # Offset by batch

                edge_src.append(src)
                edge_dst.append(dst)
                edge_attr.append(classic_rbf[b, i, j])

    # Convert lists to tensors
    pyg_x = torch.stack(pyg_nodes)
    pyg_edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    pyg_edge_attr = torch.stack(edge_attr)
    pyg_batch = torch.tensor(batch_indices, dtype=torch.long)

    print(f"Classic features shape: {classic_features.shape}")
    print(f"PyG node features shape: {pyg_x.shape}")
    print(f"PyG edge index shape: {pyg_edge_index.shape}")

    # Initialize models
    classic_model = ClassicInteractionBlock(
        n_inputs=n_features,
        n_gaussians=n_gaussians,
        n_filters=n_features,
        activation=torch.nn.Tanh()
    )

    pyg_interaction = GCNInteraction(
        in_channels=n_features,
        edge_channels=n_gaussians,
        hidden_channels=n_features,
        activation='tanh',
        use_attention=True
    )

    pyg_schnet = SchNetEncoderNoEmbed(
        node_dim=n_features,
        edge_dim=n_gaussians,
        hidden_dim=n_features,
        output_dim=n_features,
        n_interactions=1,
        activation='tanh',
        use_attention=True
    )

    # Copy weights for fair comparison
    print("\nCopying weights from classic model to PyG models...")

    # Initial dense layer
    pyg_interaction.initial_dense.weight.data = classic_model.initial_dense[0].weight.data.clone()

    # Filter network weights
    pyg_interaction.cfconv.filter_network.lins[0].weight.data = classic_model.cfconv.filter_generator[
        0].weight.data.clone()
    pyg_interaction.cfconv.filter_network.lins[0].bias.data = classic_model.cfconv.filter_generator[0].bias.data.clone()
    pyg_interaction.cfconv.filter_network.lins[1].weight.data = classic_model.cfconv.filter_generator[
        2].weight.data.clone()
    pyg_interaction.cfconv.filter_network.lins[1].bias.data = classic_model.cfconv.filter_generator[2].bias.data.clone()

    # Attention vector
    pyg_interaction.cfconv.attention_vector.data = classic_model.cfconv.nbr_filter.data.clone()

    # Output layers
    pyg_interaction.output_layer.lins[0].weight.data = classic_model.output_dense[0].weight.data.clone()
    pyg_interaction.output_layer.lins[0].bias.data = classic_model.output_dense[0].bias.data.clone()
    pyg_interaction.output_layer.lins[1].weight.data = classic_model.output_dense[2].weight.data.clone()
    pyg_interaction.output_layer.lins[1].bias.data = classic_model.output_dense[2].bias.data.clone()

    # Copy to SchNet encoder's first interaction block
    pyg_schnet.interactions[0].initial_dense.weight.data = classic_model.initial_dense[0].weight.data.clone()

    pyg_schnet.interactions[0].cfconv.filter_network.lins[0].weight.data = classic_model.cfconv.filter_generator[
        0].weight.data.clone()
    pyg_schnet.interactions[0].cfconv.filter_network.lins[0].bias.data = classic_model.cfconv.filter_generator[
        0].bias.data.clone()
    pyg_schnet.interactions[0].cfconv.filter_network.lins[1].weight.data = classic_model.cfconv.filter_generator[
        2].weight.data.clone()
    pyg_schnet.interactions[0].cfconv.filter_network.lins[1].bias.data = classic_model.cfconv.filter_generator[
        2].bias.data.clone()

    pyg_schnet.interactions[0].cfconv.attention_vector.data = classic_model.cfconv.nbr_filter.data.clone()

    pyg_schnet.interactions[0].output_layer.lins[0].weight.data = classic_model.output_dense[0].weight.data.clone()
    pyg_schnet.interactions[0].output_layer.lins[0].bias.data = classic_model.output_dense[0].bias.data.clone()
    pyg_schnet.interactions[0].output_layer.lins[1].weight.data = classic_model.output_dense[2].weight.data.clone()
    pyg_schnet.interactions[0].output_layer.lins[1].bias.data = classic_model.output_dense[2].bias.data.clone()

    print("Weight copying complete")

    # Run models
    with torch.no_grad():
        # Classic model
        classic_output, classic_attn = classic_model(classic_features, classic_rbf, classic_neighbors)
        print(f"\nClassic output shape: {classic_output.shape}")

        # PyG interaction model
        pyg_output, pyg_attn = pyg_interaction(pyg_x, pyg_edge_index, pyg_edge_attr)
        print(f"PyG interaction output shape: {pyg_output.shape}")

        # PyG SchNet model - control residual connection
        pyg_schnet.use_residual = False  # Turn off residual for direct comparison
        pyg_output_pooled, (pyg_node_features, _, pyg_deltas) = pyg_schnet(pyg_x, pyg_edge_index, pyg_edge_attr,
                                                                           pyg_batch)
        print(f"PyG SchNet node features shape: {pyg_node_features.shape}")
        print(f"PyG SchNet delta shape: {pyg_deltas[0].shape}")  # First interaction's delta

        # Get the "delta" from SchNet's first interaction for comparison
        pyg_delta = pyg_deltas[0]

    # Reshape PyG outputs to classic format for comparison
    pyg_interaction_reshaped = torch.zeros_like(classic_output)
    pyg_delta_reshaped = torch.zeros_like(classic_output)

    for b in range(batch_size):
        for i in range(n_atoms):
            idx = i + b * n_atoms
            pyg_interaction_reshaped[b, i] = pyg_output[idx]
            pyg_delta_reshaped[b, i] = pyg_delta[idx]

    # Compare outputs
    interaction_diff = torch.abs(classic_output - pyg_interaction_reshaped)
    interaction_mean_diff = interaction_diff.mean().item()
    interaction_max_diff = interaction_diff.max().item()

    delta_diff = torch.abs(classic_output - pyg_delta_reshaped)
    delta_mean_diff = delta_diff.mean().item()
    delta_max_diff = delta_diff.max().item()

    print("\n=== Classic vs PyG Interaction ===")
    print(f"Mean absolute difference: {interaction_mean_diff:.8f}")
    print(f"Maximum absolute difference: {interaction_max_diff:.8f}")

    print("\n=== Classic vs PyG SchNet Delta (before residual) ===")
    print(f"Mean absolute difference: {delta_mean_diff:.8f}")
    print(f"Maximum absolute difference: {delta_max_diff:.8f}")

    # Test with residual on
    pyg_schnet.use_residual = True
    pyg_output_pooled, (pyg_node_features_with_res, _, _) = pyg_schnet(pyg_x, pyg_edge_index, pyg_edge_attr, pyg_batch)

    # Reshape node features with residual to classic format
    pyg_with_res_reshaped = torch.zeros_like(classic_output)
    for b in range(batch_size):
        for i in range(n_atoms):
            idx = i + b * n_atoms
            pyg_with_res_reshaped[b, i] = pyg_node_features_with_res[idx]

    # Compare with classic output + input (residual connection)
    classic_with_res = classic_features + classic_output
    with_res_diff = torch.abs(classic_with_res - pyg_with_res_reshaped)
    with_res_mean_diff = with_res_diff.mean().item()
    with_res_max_diff = with_res_diff.max().item()

    print("\n=== Classic + Input vs PyG SchNet with Residual ===")
    print(f"Mean absolute difference: {with_res_mean_diff:.8f}")
    print(f"Maximum absolute difference: {with_res_max_diff:.8f}")

    return {
        'classic_output': classic_output,
        'pyg_interaction': pyg_interaction_reshaped,
        'pyg_delta': pyg_delta_reshaped,
        'interaction_mean_diff': interaction_mean_diff,
        'delta_mean_diff': delta_mean_diff,
        'with_res_mean_diff': with_res_mean_diff
    }


if __name__ == "__main__":
    try:
        results = run_debug_comparison()
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        import traceback

        traceback.print_exc()
