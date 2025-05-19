import torch
import torch.nn as nn
import torch.nn.init as init


def init_weights(model, method='kaiming_normal', nonlinearity='relu', gain=None):
    """
    Initialize model weights with different strategies, with improved bias handling
    and special handling for graph neural network layers.

    Parameters
    ----------
    model : torch.nn.Module
        The model to initialize
    method : str
        Initialization method: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
        'kaiming_normal', 'orthogonal', or 'normal'
    nonlinearity : str
        Nonlinearity for Kaiming initialization ('relu', 'leaky_relu', etc.)
    gain : float or None
        Gain factor for the initialization. If None, will use default values.

    Returns
    -------
    dict
        Report on initialization with count of parameters initialized
    """
    import torch.nn as nn
    import torch.nn.init as init
    import math

    # Set default gain values based on method
    if gain is None:
        if method.startswith('xavier'):
            gain = 1.0
        elif method.startswith('kaiming'):
            # Let Kaiming handle the gain based on nonlinearity
            gain = 0  # placeholder, not used directly
        elif method == 'orthogonal':
            gain = 1.0
        else:  # normal or uniform
            gain = 1.0

    # Track which layers were initialized
    initialized_layers = {
        'Linear': 0,
        'Conv': 0,
        'BatchNorm': 0,
        'LSTM': 0,
        'GRU': 0,
        'Embedding': 0,
        'GraphConv': 0,  # Special tracking for graph convolution layers
        'Other': 0,
        'Total params': 0,
        'Total tensors': 0
    }

    def init_tensor(tensor, layer_type):
        initialized_layers['Total tensors'] += 1
        param_count = tensor.numel()
        initialized_layers['Total params'] += param_count

        if layer_type not in initialized_layers:
            initialized_layers[layer_type] = param_count
        else:
            initialized_layers[layer_type] += param_count

        # Apply the requested initialization
        if method == 'xavier_uniform':
            init.xavier_uniform_(tensor, gain=gain)
        elif method == 'xavier_normal':
            init.xavier_normal_(tensor, gain=gain)
        elif method == 'kaiming_uniform':
            init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity=nonlinearity)
        elif method == 'kaiming_normal':
            init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity=nonlinearity)
        elif method == 'orthogonal':
            init.orthogonal_(tensor, gain=gain)
        elif method == 'normal':
            init.normal_(tensor, mean=0.0, std=0.01 * gain)
        elif method == 'uniform':
            # Calculate bounds based on gain
            bound = gain * (1.0 / tensor.size(0)) ** 0.5
            init.uniform_(tensor, -bound, bound)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

        return param_count

    # Initialize model parameters
    for name, module in model.named_modules():
        # Linear layers (including those in GNN modules)
        if isinstance(module, nn.Linear):
            init_tensor(module.weight, 'Linear')
            if module.bias is not None:
                # Custom bias initialization based on layer position in the network
                if 'output_layer' in name or 'final' in name:
                    # For output layers, use non-zero values to break symmetry
                    # This is crucial for layers with gradient problems
                    init.uniform_(module.bias, -0.1, 0.1)
                elif 'encoder' in name:
                    # For encoder layers
                    if 'lins.0' in name or '.0.' in name:  # First layer in a block
                        # Slight positive bias for first layers helps with activation
                        init.uniform_(module.bias, 0.01, 0.1)
                    else:
                        # Middle layers get smaller random initialization
                        fan_in = module.weight.size(1)
                        bound = 1 / math.sqrt(fan_in)
                        init.uniform_(module.bias, -bound, bound)
                else:
                    # Default case - small random values
                    init.uniform_(module.bias, -0.05, 0.05)

        # Convolutional layers
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init_tensor(module.weight, 'Conv')
            if module.bias is not None:
                # Small random initialization for convolution biases
                init.uniform_(module.bias, -0.05, 0.05)

        # BatchNorm layers - use standard initialization
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if module.weight is not None:
                init.constant_(module.weight, 1.0)
            if module.bias is not None:
                # Use small non-zero values for BatchNorm biases
                init.constant_(module.bias, 0.01)
            initialized_layers['BatchNorm'] += 2

        # Recurrent layers (LSTM, GRU)
        elif isinstance(module, nn.LSTM):
            for param_name, param in module.named_parameters():
                if 'weight' in param_name:
                    init_tensor(param, 'LSTM')
                elif 'bias' in param_name:
                    # Initialize forget gate bias to 1.0, others to 0.0
                    # This is a good practice for LSTM training
                    n = param.size(0)
                    forget_gate_size = n // 4
                    param.data.fill_(0.0)
                    param.data[forget_gate_size:2 * forget_gate_size].fill_(1.0)

        elif isinstance(module, nn.GRU):
            for param_name, param in module.named_parameters():
                if 'weight' in param_name:
                    init_tensor(param, 'GRU')
                elif 'bias' in param_name:
                    # Use small random values for GRU biases
                    init.uniform_(param, -0.05, 0.05)

        # Embedding layers
        elif isinstance(module, nn.Embedding):
            # Use a smaller initialization for embeddings
            init.normal_(module.weight, mean=0.0, std=0.01)
            initialized_layers['Embedding'] += module.weight.numel()

        # Special handling for graph neural network layers
        elif any(graph_layer in module.__class__.__name__ for graph_layer in
                 ['GCNConv', 'GATConv', 'SAGEConv', 'GINConv', 'TransformerConv']):
            # For graph neural network specific modules
            graph_params = 0
            for param_name, param in module.named_parameters():
                if 'weight' in param_name:
                    if method == 'kaiming_normal':
                        init.kaiming_normal_(param, a=0, mode='fan_out', nonlinearity=nonlinearity)
                    elif method == 'xavier_normal':
                        init.xavier_normal_(param, gain=gain)
                    else:
                        init_tensor(param, 'GraphConv')
                    graph_params += param.numel()
                elif 'bias' in param_name:
                    # Add special handling for biases in attention-based layers
                    if 'attention' in param_name.lower() or 'attn' in param_name.lower():
                        # For attention-based layers, non-zero bias is important
                        init.uniform_(param, 0.01, 0.1)
                    else:
                        # For other GNN biases
                        if param.dim() > 0:  # Make sure it's not empty
                            init.uniform_(param, -0.1, 0.1)
                    graph_params += param.numel()

            initialized_layers['GraphConv'] += graph_params

    return initialized_layers


def init_for_vamp(model, method='kaiming_normal'):
    """
    Initialize weights optimized for VAMPNet models with graph neural networks.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to initialize
    method : str, optional
        Initialization method, default 'kaiming_normal'
    """
    # Check if this looks like a graph neural network
    is_graph_model = False
    for name, module in model.named_modules():
        if any(gnn_type in str(type(module)) for gnn_type in
               ['GCN', 'GAT', 'GraphConv', 'GIN', 'EdgeConv']):
            is_graph_model = True
            break

    # Use method appropriate for the model type
    if is_graph_model:
        # For GNNs, use fan_out mode with kaiming_normal
        if method == 'kaiming_normal':
            init_report = init_weights(model, method='kaiming_normal', nonlinearity='relu')
        elif method == 'orthogonal':
            # Orthogonal often works well for GNN problems
            init_report = init_weights(model, method='orthogonal', gain=0.8)
        else:
            init_report = init_weights(model, method=method)
    else:
        # Standard initialization for non-graph models
        if method == 'kaiming_normal':
            init_report = init_weights(model, method='kaiming_normal', nonlinearity='relu')
        elif method == 'xavier_normal':
            init_report = init_weights(model, method='xavier_normal', gain=0.8)  # Lower gain than before
        else:
            init_report = init_weights(model, method=method)

    # Print initialization report
    print(f"\nWeight Initialization Report ({method}):")
    print("-" * 40)
    for layer_type, count in sorted(init_report.items()):
        if layer_type == 'Total params':
            print("-" * 40)
        if 'params' in layer_type or 'tensors' in layer_type:
            print(f"{layer_type}: {count:,}")
        elif count > 0:
            print(f"{layer_type} parameters: {count:,}")
    print("-" * 40)

    # Check for uninitialized parameters
    uninit_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        # Check if parameter looks uninitialized (near zero or constant)
        if param.requires_grad and (param.std() < 1e-6 or param.min() == param.max()):
            print(f"WARNING: Parameter possibly uninitialized: {name}, shape: {param.shape}")
            uninit_params += 1

    if uninit_params > 0:
        print(f"WARNING: Found {uninit_params}/{total_params} parameters that may not be properly initialized.")

    return init_report


def monitor_gradients(model, epoch, batch_idx=0, log_interval=1):
    """
    Monitor gradients in a PyTorch model to detect vanishing or exploding gradients.

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model to monitor
    epoch : int
        Current epoch number
    batch_idx : int
        Current batch index
    log_interval : int
        How often to log detailed information

    Returns:
    --------
    dict
        Statistics about gradients
    """
    if epoch % log_interval != 0 and batch_idx != 0:
        return None

    grad_stats = {
        'max': 0.0,
        'min': 0.0,
        'mean': 0.0,
        'median': 0.0,
        'std': 0.0,
        'zero_percent': 0.0,
        'large_percent': 0.0,  # > 1.0
        'small_percent': 0.0,  # < 1e-4
        'layer_norms': {},
        'layer_means': {},
        'layer_maxes': {},
    }

    # Collect gradients from all parameters
    all_grads = []
    total_params = 0
    zero_grads = 0
    large_grads = 0
    small_grads = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            # Get gradient tensor
            grad = param.grad.detach()

            # Calculate norm for this layer
            layer_norm = torch.norm(grad).item()
            grad_stats['layer_norms'][name] = layer_norm

            # Calculate mean and max for this layer
            grad_stats['layer_means'][name] = grad.abs().mean().item()
            grad_stats['layer_maxes'][name] = grad.abs().max().item()

            # Flatten gradient for overall statistics
            grad_flat = grad.flatten()
            all_grads.append(grad_flat)

            # Count parameters and special cases
            total_params += grad_flat.numel()
            zero_grads += (grad_flat == 0).sum().item()
            large_grads += (grad_flat.abs() > 1.0).sum().item()
            small_grads += (grad_flat.abs() < 1e-4).sum().item()

    if all_grads:
        # Concatenate all gradients
        all_grads_tensor = torch.cat(all_grads)

        # Calculate overall statistics
        grad_stats['max'] = all_grads_tensor.abs().max().item()
        grad_stats['min'] = all_grads_tensor.abs().min().item()
        grad_stats['mean'] = all_grads_tensor.abs().mean().item()
        grad_stats['median'] = all_grads_tensor.abs().median().item()
        grad_stats['std'] = all_grads_tensor.std().item()

        # Calculate percentages
        grad_stats['zero_percent'] = 100 * zero_grads / total_params
        grad_stats['large_percent'] = 100 * large_grads / total_params
        grad_stats['small_percent'] = 100 * small_grads / total_params

        # Log detailed information
        if epoch % log_interval == 0 and batch_idx == 0:
            print(f"\n===== Gradient Analysis (Epoch {epoch}) =====")
            print(f"Overall gradient statistics:")
            print(f"  Max grad: {grad_stats['max']:.6f}")
            print(f"  Mean grad: {grad_stats['mean']:.6f}")
            print(f"  Median grad: {grad_stats['median']:.6f}")
            print(f"  Zero grads: {grad_stats['zero_percent']:.2f}%")
            print(f"  Large grads (>1.0): {grad_stats['large_percent']:.2f}%")
            print(f"  Small grads (<1e-4): {grad_stats['small_percent']:.2f}%")

            # Sort layers by gradient norm and show top/bottom layers
            sorted_layers = sorted(grad_stats['layer_norms'].items(), key=lambda x: x[1], reverse=True)

            print("\nLayers with largest gradient norms:")
            for name, norm in sorted_layers[:5]:
                print(f"  {name}: {norm:.6f} (max: {grad_stats['layer_maxes'][name]:.6f})")

            print("\nLayers with smallest gradient norms:")
            for name, norm in sorted_layers[-5:]:
                print(f"  {name}: {norm:.6f} (max: {grad_stats['layer_maxes'][name]:.6f})")

            print("=====================================")

    return grad_stats

