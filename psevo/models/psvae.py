#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from psevo.encoder.gin import GINEEncoder
from psevo.encoder.cheb import ChebEncoder
from psevo.encoder.ml3 import GNNML3Model as ML3Encoder
from psevo.predictor.predictor import Predictor
from psevo.decoder.vae_piecewise import VAEPieceDecoder

PROPS = ['qed', 'sa', 'logp']

class PSVAEModel(nn.Module):
    """ Complete Piece-based Structural Variational Autoencoder (PS-VAE) Model.
    This model implements the full PS-VAE architecture from the original paper with:
    1. Configurable Graph Neural Network Encoder: Choose from GIN, Chebyshev, or ML3
    2. VAE Piece Decoder: Generates molecules through piece-based reconstruction
    3. Property Predictor: Predicts molecular properties from latent space

    The model implements a complete molecular generation pipeline with:
    - Graph-level encoding of molecular structure using different GNN architectures
    - Latent space learning for molecular properties
    - Two-stage decoding: piece generation + edge prediction
    - Multi-task learning: reconstruction + property prediction

    Supported Encoders:
    - "gin": Graph Isomorphism Network with Edge features (1-WL equivalent)
    - "cheb": Chebyshev Spectral Graph Convolution (spectral domain)
    - "ml3": GNNML3 with 3-WL expressive power (most powerful)

    Mathematical Framework:
    - Graph Embedding: h_graph = Encoder(embed(x, pieces, pos), edges)
    - VAE Loss: L_vae = L_recon + β * L_KL
    - Property Loss: L_prop = MSE(predictor(z), properties)
    - Total Loss: L = α * L_recon + (1-α) * L_prop + β * L_KL
    """

    def __init__(self, config, tokenizer):
        super(PSVAEModel, self).__init__()

        # Store configuration and tokenizer
        self.config = config
        self.tokenizer = tokenizer
        self.chem_vocab = tokenizer.chem_vocab

        # Extract encoder configuration
        encoder_config = config['encoder']
        encoder_type = encoder_config.get('type', 'gin').lower()

        # Validate encoder type
        if encoder_type not in ['gin', 'cheb', 'ml3']:
            raise ValueError(f"Unsupported encoder type: {encoder_type}. "
                             f"Supported types: ['gin', 'cheb', 'ml3']")

        # Initialize configurable encoder based on type
        self.encoder_type = encoder_type
        self.encoder = self._create_encoder(encoder_type, encoder_config)

        # Initialize other components
        self.predictor = Predictor(**(config['predictor']))
        self.decoder = VAEPieceDecoder(**(config['vae_piece_decoder']), tokenizer=tokenizer)

        # Loss functions
        self.pred_loss = nn.MSELoss()  # Property prediction loss

        # Training metrics
        self.total_time = 0

    def _create_encoder(self, encoder_type, encoder_config):
        """
        Create the appropriate encoder based on the specified type.

        Args:
            encoder_type (str): Type of encoder ('gin', 'cheb', 'ml3')
            encoder_config (dict): Configuration parameters for the encoder

        Returns:
            nn.Module: Initialized encoder module
        """
        # Extract common parameters
        dim_in = encoder_config['dim_in']
        dim_hidden = encoder_config['dim_hidden']
        dim_out = encoder_config['dim_out']

        if encoder_type == "gin":
            # Graph Isomorphism Network with Edge features
            # 1-WL equivalent, good for edge-conditioned tasks
            return GINEEncoder(
                dim_in=dim_in,
                num_edge_type=encoder_config['num_edge_type'],
                dim_hidden=dim_hidden,
                dim_out=dim_out,
                t=encoder_config.get('t', 4)  # Number of message passing iterations
            ).to("cuda")

        elif encoder_type == "cheb":
            # Chebyshev Spectral Graph Convolution
            # Spectral domain design, good for frequency-based tasks
            return ChebEncoder(
                dim_in=dim_in,
                num_edge_type=encoder_config['num_edge_type'],
                dim_hidden=dim_hidden,
                dim_out=dim_out,
                t=encoder_config.get('t', 4),  # Number of message passing iterations
                K=encoder_config.get('K', 3)  # Chebyshev filter size
            ).to("cuda")

        elif encoder_type == "ml3":
            # GNNML3 with 3-WL expressive power
            # Most powerful, can distinguish 1-WL equivalent graphs
            return ML3Encoder(
                dim_in=dim_in,
                dim_out=dim_out,
                dim_hidden=dim_hidden,
                num_layers=encoder_config.get('t', 4),  # Number of layers
                num_supports=encoder_config.get('num_supports', 5),  # Spectral supports
                bandwidth=encoder_config.get('bandwidth', 5.0),  # Spectral bandwidth
                use_adjacency=encoder_config.get('use_adjacency', False)  # Use adjacency vs Laplacian
            ).to("cuda")

    def forward(self, batch, return_accu=False):
        """
        Forward pass of PS-VAE model implementing end-to-end molecular generation.

        Architecture Flow:
        1. Multi-modal embedding: Combine atom, piece, and positional features
        2. Graph encoding: Learn structural representations via configurable GNN
        3. VAE decoding: Generate pieces and predict edges
        4. Property prediction: Predict molecular properties from latent space
        5. Multi-task loss: Combine reconstruction and prediction losses

        Args:
            batch (dict): Batch data containing molecular graph information
            return_accu (bool): Whether to return accuracy metrics

        Returns:
            tuple: (property_loss, vae_results)
                - property_loss: MSE loss for property prediction
                - vae_results: VAE decoder outputs (losses and optionally accuracies)
        """

        # =================================================================
        # STAGE 1: MULTI-MODAL NODE EMBEDDING
        # =================================================================

        # Extract batch components
        x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr']
        x_pieces, x_pos = batch['x_pieces'], batch['x_pos']

        # Create rich node representations by combining multiple modalities
        x = self.decoder.embed_atom(x, x_pieces, x_pos)  # [batch_size, num_nodes, embed_dim]

        # =================================================================
        # STAGE 2: CONFIGURABLE GRAPH NEURAL NETWORK ENCODING
        # =================================================================

        batch_size, node_num, node_dim = x.shape

        # Create graph assignment vector for batch processing
        graph_ids = torch.repeat_interleave(
            torch.arange(0, batch_size, device=x.device),
            node_num
        )  # [batch_size * num_nodes]

        # Reshape node features for GNN processing
        node_x = x.view(-1, node_dim)  # [batch_size * num_nodes, node_dim]

        # Apply configurable graph neural network based on encoder type
        if self.encoder_type in ['gin', 'cheb']:
            # Traditional MPNN encoders (GIN, Chebyshev)
            # Use embed_node method for node-level representations
            _, all_x = self.encoder.embed_node(node_x, edge_index, edge_attr)
            # all_x: [batch_size * num_nodes, hidden_dim * num_iterations]

            # Aggregate node embeddings to create graph-level representations
            graph_embedding = self.encoder.embed_graph(
                all_x,
                graph_ids,
                batch['atom_mask'].flatten()
            )  # [batch_size, graph_feature_dim]

        elif self.encoder_type == 'ml3':
            # GNNML3 encoder with different interface
            # Create temporary data object for ML3 encoder
            ml3_data = type('Data', (), {})()
            ml3_data.x = node_x
            ml3_data.edge_index = edge_index
            ml3_data.batch = graph_ids
            ml3_data.num_graphs = batch_size

            # ML3 encoder returns graph-level embeddings directly
            graph_embedding = self.encoder(ml3_data)  # [batch_size, graph_feature_dim]

        # =================================================================
        # STAGE 3: VAE PIECE-BASED DECODING
        # =================================================================

        # Filter edges to only include intra-piece connections
        in_piece_edge_idx = batch['in_piece_edge_idx']
        filtered_edge_index = edge_index[:, in_piece_edge_idx]
        filtered_edge_attr = edge_attr[in_piece_edge_idx]

        # VAE decoding with two-stage generation
        z, vae_results = self.decoder(
            x=x,  # Multi-modal node embeddings
            x_pieces=x_pieces,  # Piece type indices
            x_pos=x_pos,  # Position indices
            edge_index=filtered_edge_index,  # Intra-piece edges only
            edge_attr=filtered_edge_attr,  # Intra-piece edge features
            pieces=batch['pieces'],  # Target piece sequences
            conds=graph_embedding,  # Graph-level conditioning
            edge_select=batch['edge_select'],  # Inter-piece edge prediction mask
            golden_edge=batch['golden_edge'],  # Target edge types
            return_accu=return_accu  # Whether to return accuracies
        )

        # =================================================================
        # STAGE 4: MOLECULAR PROPERTY PREDICTION
        # =================================================================

        # Predict molecular properties from latent representations
        pred_prop = self.predictor(z)  # [batch_size, num_properties]

        # Extract target properties for selected property indices
        golden_props = batch['props'].reshape(batch_size, -1)
        selected_props = golden_props[:, self.config['selected_properties']]
        selected_props = selected_props.float()

        # Compute property prediction loss (MSE)
        property_loss = self.pred_loss(pred_prop, selected_props)

        return property_loss, vae_results

    def cal_beta(self, global_step):
        """
        Calculate KL annealing beta parameter (from original code).

        Implements the annealing schedule:
        - Warmup period: beta = 0
        - Annealing period: beta increases stepwise
        - Final period: beta = max_beta

        Args:
            global_step (int): Current training step

        Returns:
            float: Current beta value for KL loss weighting
        """
        step = global_step
        warmup = self.config.get('kl_warmup', 0)

        if step < warmup:
            beta = 0
        else:
            step_beta = self.config.get('step_beta', 0.002)
            anneal_iter = self.config.get('kl_anneal_iter', 1000)
            max_beta = self.config.get('max_beta', 0.01)
            beta = min(max_beta, ((step - warmup) // anneal_iter + 1) * step_beta)

        beta += self.config.get('beta', 0)
        return beta

    def weighted_loss(self, pred_loss, rec_loss, kl_loss, global_step):
        """
        Combine losses with learned weighting (from original code).

        Formula: α * rec_loss + (1 - α) * pred_loss + β * kl_loss

        Args:
            pred_loss (torch.Tensor): Property prediction loss
            rec_loss (torch.Tensor): Reconstruction loss
            kl_loss (torch.Tensor): KL divergence loss
            global_step (int): Current training step

        Returns:
            tuple: (total_loss, beta_value)
        """
        alpha = self.config.get('alpha', 0.5)
        beta = self.cal_beta(global_step)

        total_loss = alpha * rec_loss + (1 - alpha) * pred_loss + beta * kl_loss
        return total_loss, beta

    def training_step(self, batch, optimizer, global_step):
        """
        Training step for the PS-VAE model with configurable encoder.

        Args:
            batch (dict): Training batch data
            optimizer: PyTorch optimizer
            global_step (int): Current global training step

        Returns:
            dict: Dictionary containing loss components and metrics
        """
        # Forward pass
        pred_loss, vae_results = self.forward(batch, return_accu=True)

        # Extract VAE loss components
        (rec_loss, piece_loss, edge_loss, kl_loss), (piece_accu, edge_accu) = vae_results

        # Compute weighted loss using original formula
        total_loss, beta = self.weighted_loss(pred_loss, rec_loss, kl_loss, global_step)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping (from original config)
        if 'grad_clip' in self.config:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config['grad_clip'])

        optimizer.step()

        # Return metrics with encoder type information
        return {
            'total_loss': total_loss.item(),
            'pred_loss': pred_loss.item(),
            'rec_loss': rec_loss.item(),
            'piece_loss': piece_loss.item(),
            'edge_loss': edge_loss.item(),
            'kl_loss': kl_loss.item(),
            'piece_accuracy': piece_accu,
            'edge_accuracy': edge_accu,
            'beta': beta,
            'encoder_type': self.encoder_type
        }

    def validation_step(self, batch, global_step):
        """Validation step for the PS-VAE model."""
        self.eval()
        with torch.no_grad():
            pred_loss, vae_results = self.forward(batch, return_accu=True)
            (rec_loss, piece_loss, edge_loss, kl_loss), (piece_accu, edge_accu) = vae_results
            total_loss, beta = self.weighted_loss(pred_loss, rec_loss, kl_loss, global_step)

        self.train()

        return {
            'val_total_loss': total_loss.item(),
            'val_pred_loss': pred_loss.item(),
            'val_rec_loss': rec_loss.item(),
            'val_piece_loss': piece_loss.item(),
            'val_edge_loss': edge_loss.item(),
            'val_kl_loss': kl_loss.item(),
            'val_piece_accuracy': piece_accu,
            'val_edge_accuracy': edge_accu,
            'val_beta': beta,
            'encoder_type': self.encoder_type
        }

    def generate_molecule(self, properties, max_atom_num=50, add_edge_th=0.5, temperature=1.0):
        """Generate a molecule from target properties."""
        self.eval()
        with torch.no_grad():
            properties = properties.unsqueeze(0)
            z_mean = self.decoder.W_mean(properties)
            z = z_mean
            molecule = self.decoder.inference(z.squeeze(0), max_atom_num, add_edge_th, temperature)
        self.train()
        return molecule

    def generate_molecule_no_vae(self, latent_vec, max_atom_num=50, add_edge_th=0.5, temperature=1.0):
        """Generate a molecule from latent embedding without the VAE approach."""
        self.eval()
        with torch.no_grad():
            molecule = self.decoder.inference(latent_vec, max_atom_num, add_edge_th, temperature)
        self.train()
        return molecule

    def generate_molecule_with_constraint(self, properties, constraint_mol,
                                          max_atom_num=50, add_edge_th=0.5, temperature=1.0):
        """Generate a molecule from target properties with structural constraints."""
        self.eval()
        with torch.no_grad():
            properties = properties.unsqueeze(0)
            z_mean = self.decoder.W_mean(properties)
            z = z_mean
            molecule = self.decoder.inference_constraint(
                z.squeeze(0), max_atom_num, add_edge_th, temperature, constraint_mol
            )
        self.train()
        return molecule

    def sample_z(self, n, device=None):
        """Sample random latent vectors from standard normal distribution."""
        if device is not None:
            return torch.randn(n, self.decoder.latent_dim, device=device)
        return torch.randn(n, self.decoder.latent_dim, device=next(self.parameters()).device)

    def get_z(self, batch):
        """Encode a batch of molecules to latent space."""
        x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr']
        x_pieces, x_pos = batch['x_pieces'], batch['x_pos']
        x = self.decoder.embed_atom(x, x_pieces, x_pos)
        batch_size, node_num, node_dim = x.shape
        graph_ids = torch.repeat_interleave(torch.arange(0, batch_size, device=x.device), node_num)

        # Apply encoder based on type
        if self.encoder_type in ['gin', 'cheb']:
            _, all_x = self.encoder.embed_node(x.view(-1, node_dim), edge_index, edge_attr)
            graph_embedding = self.encoder.embed_graph(all_x, graph_ids, batch['atom_mask'].flatten())
        elif self.encoder_type == 'ml3':
            ml3_data = type('Data', (), {})()
            ml3_data.x = x.view(-1, node_dim)
            ml3_data.edge_index = edge_index
            ml3_data.batch = graph_ids
            ml3_data.num_graphs = batch_size
            graph_embedding = self.encoder(ml3_data)

        z_vecs, _ = self.decoder.rsample(graph_embedding)
        return z_vecs

    def predict_props(self, z):
        """Predict molecular properties from latent vectors."""
        props = self.predictor(z)
        # Handle padding for evaluation (from original code)
        if len(props.shape) == 1:
            # Single sample - pad to full property vector
            padded_props = [torch.tensor(0, device=props.device) for _ in PROPS]
            for i, val in zip(self.config['selected_properties'], props):
                padded_props[i] = val
            return torch.stack(padded_props)
        else:
            # Batch - pad to full property matrix
            padded_props = torch.zeros(props.shape[0], len(PROPS), device=props.device)
            prop_ids = self.config['selected_properties']
            padded_props[:, prop_ids] = props
            return padded_props

    def get_encoder_info(self):
        """Get information about the current encoder configuration."""
        return {
            'type': self.encoder_type,
            'parameters': sum(p.numel() for p in self.encoder.parameters()),
            'description': {
                'gin': 'Graph Isomorphism Network with Edge features (1-WL equivalent)',
                'cheb': 'Chebyshev Spectral Graph Convolution (spectral domain)',
                'ml3': 'GNNML3 with 3-WL expressive power (most powerful)'
            }[self.encoder_type]
        }
