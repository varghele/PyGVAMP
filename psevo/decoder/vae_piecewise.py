import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MLP

from rdkit import Chem

from psevo.encoder.gin import GINEEncoder
from psevo.encoder.cheb import ChebEncoder
from psevo.encoder.ml3 import GNNML3Model as ML3Encoder

from psevo.utils.chem_utils import smiles2molecule, valence_check, cnt_atom, cycle_check


class VAEPieceDecoder(nn.Module):
    """ Variational Autoencoder Piece Decoder for molecular generation.
    This decoder generates molecules in two stages:
    1. Sequential piece generation using RNN (molecular fragments/pieces)
    2. Edge prediction between atoms using graph neural networks

    Mathematical Background:
    - Uses reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)
    - KL divergence: D_KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    - Reconstruction loss: piece prediction + edge prediction losses
    """

    def __init__(self,
                 atom_embedding_dim,
                 piece_embedding_dim,
                 max_pos,
                 pos_embedding_dim,
                 piece_hidden_dim,
                 node_hidden_dim,
                 num_edge_type,
                 cond_dim,
                 latent_dim,
                 tokenizer,
                 t=4,
                 encoder_type="gin"):
        """
        Initialize the VAE Piece Decoder.

        Args:
            atom_embedding_dim (int): Dimension for atom type embeddings
            piece_embedding_dim (int): Dimension for molecular piece embeddings
            max_pos (int): Maximum position index for positional encoding
            pos_embedding_dim (int): Dimension for positional embeddings
            piece_hidden_dim (int): Hidden dimension for RNN piece generation
            node_hidden_dim (int): Hidden dimension for graph node representations
            num_edge_type (int): Number of different bond types (single, double, triple, etc.)
            cond_dim (int): Dimension of conditioning vector (molecular properties)
            latent_dim (int): Dimension of VAE latent space
            tokenizer: Molecular tokenizer for piece vocabulary
            t (int): Number of message passing iterations in graph encoder
            encoder_type (str): Type of encoder to use ("cheb", "gin", "ml3")
        """
        super(VAEPieceDecoder, self).__init__()

        self.tokenizer = tokenizer

        # =================================================================
        # PIECE PREDICTION COMPONENTS (Sequential Generation)
        # =================================================================

        # Embedding layers for different molecular components
        self.atom_embedding = nn.Embedding(
            tokenizer.num_atom_type(),
            atom_embedding_dim
        )  # Maps atom types (C, N, O, etc.) to dense vectors

        self.piece_embedding = nn.Embedding(
            tokenizer.num_piece_type(),
            piece_embedding_dim
        )  # Maps molecular pieces/fragments to dense vectors

        self.pos_embedding = nn.Embedding(
            max_pos,
            pos_embedding_dim
        )  # Positional encoding for sequence order (max position = 99, 0 = padding)

        # VAE latent to RNN hidden state transformation
        self.latent_to_rnn_hidden = nn.Linear(latent_dim, piece_hidden_dim)

        # Recurrent neural network for sequential piece generation
        # Takes piece embeddings as input, outputs hidden states for vocabulary prediction
        self.rnn = nn.GRU(
            piece_embedding_dim,
            piece_hidden_dim,
            batch_first=True
        )

        # Output layer: hidden states → piece vocabulary probabilities
        self.to_vocab = nn.Linear(piece_hidden_dim, tokenizer.num_piece_type())

        # =================================================================
        # GRAPH EMBEDDING COMPONENTS (Structural Representation)
        # =================================================================

        # Combined node feature dimension: atom + piece + position information
        node_dim = atom_embedding_dim + piece_embedding_dim + pos_embedding_dim

        # Graph neural network encoder for learning node representations
        # Uses message passing to capture local molecular structure
        if encoder_type == "gin":
            self.graph_embedding = GINEEncoder(
                dim_in=node_dim,  # feature dimension for nodes
                num_edge_type=num_edge_type,  # edge feature dimension
                dim_hidden=node_hidden_dim,
                dim_out=1,  # dim_out not used in this context, dimension for final graph embeddings
                t=t, # Number of message passing iterations
            )
        elif encoder_type == "cheb":
            self.graph_embedding = ChebEncoder(
                dim_in=node_dim,  # feature dimension for nodes
                num_edge_type=num_edge_type,  # edge feature dimension
                dim_hidden=node_hidden_dim,
                dim_out=1,  # dim_out not used in this context, dimension for final graph embeddings
                t=t,  # Number of message passing iterations
                K=3,  # Chebyshev filter size
            )
        elif encoder_type == "ml3":
            self.graph_embedding = ML3Encoder(
                dim_in=node_dim,
                dim_hidden=node_hidden_dim,
                dim_out=1,
                num_layers=t,
                num_supports=num_edge_type,
                bandwidth=5.0,
                use_adjacency=False,
            )

        # =================================================================
        # EDGE LINK PREDICTION COMPONENTS (Bond Formation)
        # =================================================================

        # Input dimension for edge predictor:
        # source_node + target_node + latent_context
        mlp_in = node_hidden_dim * 2 + latent_dim

        # Multi-layer perceptron for predicting bond types between atom pairs
        # Architecture: MLP → Linear → bond_type_probabilities
        self.edge_predictor = MLP(
            in_channels=mlp_in,
            hidden_channels=mlp_in // 2,  # Bottleneck architecture
            out_channels=num_edge_type,
            num_layers=4,
            act="relu",
            plain_last=True, # Final classification layer is without act
        )


        # =================================================================
        # VARIATIONAL AUTOENCODER COMPONENTS
        # =================================================================

        self.latent_dim = latent_dim

        # VAE encoder outputs: condition → (mean, log_variance)
        # Reparameterization trick: z = mean + exp(log_var/2) * epsilon
        self.W_mean = nn.Linear(cond_dim, latent_dim)      # μ(condition)
        self.W_log_var = nn.Linear(cond_dim, latent_dim)   # log(σ²(condition))

        # =================================================================
        # LOSS FUNCTIONS
        # =================================================================

        # Cross-entropy loss for piece sequence prediction
        # Ignores padding tokens in loss calculation
        self.piece_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx())

        # Cross-entropy loss for edge/bond type prediction
        self.edge_loss = nn.CrossEntropyLoss()

    def rsample(self, conds):
        """
        Reparameterization sampling for VAE latent space.

        Mathematical Background:
        Implements the reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)
        This allows gradients to flow through the sampling process during training.

        KL Divergence Calculation:
        D_KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        Where q(z|x) is the encoder distribution and p(z) = N(0,I) is the prior.

        Args:
            conds (Tensor): Conditioning vector (molecular properties) [batch_size, cond_dim]

        Returns:
            tuple: (z_vecs, kl_loss)
                - z_vecs: Sampled latent vectors [batch_size, latent_dim]
                - kl_loss: KL divergence loss (scalar)
        """
        batch_size = conds.shape[0]

        # Encode conditions to latent distribution parameters
        z_mean = self.W_mean(conds)  # μ(condition)
        z_log_var = -torch.abs(self.W_log_var(conds))  # log(σ²(condition))

        # Note: Following Mueller et al. - ensure log_var is negative for stability
        # This constrains σ² ∈ (0,1], preventing numerical issues

        # Compute KL divergence: D_KL(q(z|x) || N(0,I))
        # Formula: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(
            1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)
        ) / batch_size

        # Reparameterization trick: z = μ + σ * ε
        epsilon = torch.randn_like(z_mean)  # ε ~ N(0,1)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon  # z = μ + σ * ε

        return z_vecs, kl_loss

    def embed_atom(self, atom_ids, piece_ids, pos_ids):
        """ Create multi-modal atom embeddings by combining different feature types.

        Args:
            atom_ids (Tensor): Atom type indices [batch_size, num_atoms]
            piece_ids (Tensor): Piece type indices [batch_size, num_atoms]
            pos_ids (Tensor): Position indices [batch_size, num_atoms]

        Returns:
            Tensor: Concatenated embeddings [batch_size, num_atoms, total_embed_dim]
                    where total_embed_dim = atom_embed + piece_embed + pos_embed
        """
        # Embed different modalities
        atom_embed = self.atom_embedding(atom_ids)      # Chemical element embeddings
        piece_embed = self.piece_embedding(piece_ids)   # Molecular piece embeddings
        pos_embed = self.pos_embedding(pos_ids)         # Positional embeddings

        # Concatenate all embedding types to create rich atom representations
        # This multi-modal approach captures both chemical and structural information
        return torch.cat([atom_embed, piece_embed, pos_embed], dim=-1)

    def forward(self, x, x_pieces, x_pos, edge_index, edge_attr, pieces, conds, edge_select, golden_edge, return_accu=False):
        """ Forward pass of VAE Piece Decoder implementing two-stage molecular generation.
        Architecture Overview:
        Stage 1: Sequential piece generation using RNN
        Stage 2: Edge prediction using graph neural networks

        Mathematical Framework:
        - VAE Loss: L = L_recon + β * L_KL
        - L_recon = L_piece + L_edge (reconstruction of pieces and edges)
        - L_KL = KL divergence between q(z|condition) and p(z) = N(0,I)

        Args:
            x (Tensor): Node features [batch_size, num_nodes, node_dim]
            x_pieces (Tensor): Piece IDs for each node [batch_size, num_nodes]
            x_pos (Tensor): Position IDs for each node [batch_size, num_nodes]
            edge_index (Tensor): Existing edge connectivity [2, num_existing_edges]
            edge_attr (Tensor): Existing edge features [num_existing_edges, edge_dim]
            pieces (Tensor): Target piece sequence [batch_size, seq_len]
            conds (Tensor): Conditioning vector [batch_size, cond_dim]
            edge_select (Tensor): Edge prediction mask [batch_size, num_nodes, num_nodes]
            golden_edge (Tensor): Target edge types [num_edges_to_predict]
            return_accu (bool): Whether to return accuracy metrics

        Returns:
            tuple: (z_vecs, loss_info)
                - z_vecs: Latent vectors [batch_size, latent_dim]
                - loss_info: Loss tuple or (loss_tuple, accuracy_tuple)
        """

        # =================================================================
        # STAGE 1: VAE LATENT SAMPLING
        # =================================================================

        # Sample from latent space using reparameterization trick
        z_vecs, kl_loss = self.rsample(conds)  # [batch_size, latent_dim]

        # =================================================================
        # STAGE 2: SEQUENTIAL PIECE GENERATION
        # =================================================================

        # Prepare targets for piece prediction (teacher forcing)
        # Shift sequence: input = pieces[:-1], target = pieces[1:]
        gold_piece = pieces[:, 1:].flatten()                    # Target pieces (flattened)
        pieces_embedded = self.piece_embedding(pieces)          # [batch_size, seq_len, embed_dim]

        # Initialize RNN hidden state from latent vector
        # This conditions the sequence generation on the sampled latent code
        init_hidden = self.latent_to_rnn_hidden(z_vecs).unsqueeze(0)  # [1, batch_size, hidden_dim]

        # Teacher forcing: use ground truth pieces as input (except last token)
        pieces_input = pieces_embedded[:, :-1]                  # [batch_size, seq_len-1, embed_dim]

        # RNN forward pass for piece sequence generation
        rnn_output, _ = self.rnn(pieces_input, init_hidden)     # [batch_size, seq_len-1, hidden_dim]

        # Convert RNN hidden states to vocabulary probabilities
        piece_logits = self.to_vocab(rnn_output)               # [batch_size, seq_len-1, vocab_size]

        # =================================================================
        # STAGE 3: GRAPH NEURAL NETWORK EMBEDDING
        # =================================================================

        # Reshape node features for GNN processing
        batch_size, node_num, node_dim = x.shape
        node_x = x.view(-1, node_dim)                          # [batch_size * num_nodes, node_dim]

        # Apply graph neural network to learn structural representations
        # This captures local molecular structure through message passing
        node_embedding, _ = self.graph_embedding.embed_node(
            node_x, edge_index, edge_attr
        )  # [batch_size * num_nodes, node_hidden_dim]

        # Reshape back to batch format
        node_embedding = node_embedding.view(batch_size, node_num, -1)  # [batch_size, num_nodes, hidden_dim]

        # =================================================================
        # STAGE 4: EDGE PREDICTION PREPARATION
        # =================================================================

        # Create all possible node pair combinations for edge prediction
        # This is computationally expensive but necessary for complete edge prediction

        # Source node embeddings: repeat each node embedding for all possible targets
        # Shape: [batch_size, num_nodes, num_nodes, hidden_dim]
        src_embedding = torch.repeat_interleave(
            node_embedding, node_num, dim=1
        ).view(batch_size, node_num, node_num, -1)
        src_embedding = src_embedding[edge_select]             # [num_edges_to_predict, hidden_dim]

        # Target node embeddings: repeat each node embedding for all possible sources
        # Shape: [batch_size, num_nodes, num_nodes, hidden_dim]
        dst_embedding = torch.repeat_interleave(
            node_embedding, node_num, dim=0
        ).view(batch_size, node_num, node_num, -1)
        dst_embedding = dst_embedding[edge_select]             # [num_edges_to_predict, hidden_dim]

        # Latent context: repeat latent vector for each potential edge
        # This conditions edge prediction on the global molecular context
        latent_repeat = torch.repeat_interleave(
            z_vecs, node_num ** 2, dim=0
        ).view(batch_size, node_num, node_num, -1)
        latent_repeat = latent_repeat[edge_select]             # [num_edges_to_predict, latent_dim]

        # =================================================================
        # STAGE 5: EDGE TYPE PREDICTION
        # =================================================================

        # Concatenate source, target, and latent features for edge prediction
        # This rich representation enables accurate bond type prediction
        edge_pred_input = torch.cat([
            src_embedding,    # Source atom features
            dst_embedding,    # Target atom features
            latent_repeat     # Global molecular context
        ], dim=-1)  # [num_edges_to_predict, 2*hidden_dim + latent_dim]

        # Predict edge types using MLP
        pred_edge_logits = self.edge_predictor(edge_pred_input)  # [num_edges_to_predict, num_edge_types]

        # =================================================================
        # STAGE 6: LOSS COMPUTATION
        # =================================================================

        # Piece prediction loss (cross-entropy with padding ignored)
        pred_piece_flat = piece_logits.view(-1, piece_logits.shape[-1])  # [batch_size * seq_len, vocab_size]
        piece_loss = self.piece_loss(pred_piece_flat, gold_piece)

        # Edge prediction loss (cross-entropy)
        edge_loss = self.edge_loss(pred_edge_logits, golden_edge)

        # Total reconstruction loss
        rec_loss = piece_loss + edge_loss

        # Complete loss tuple: (total_recon, piece_recon, edge_recon, kl_divergence)
        loss_tuple = (rec_loss, piece_loss, edge_loss, kl_loss)

        # =================================================================
        # STAGE 7: ACCURACY COMPUTATION (OPTIONAL)
        # =================================================================

        if return_accu:
            # Piece prediction accuracy (excluding padding tokens)
            not_pad = (gold_piece != self.tokenizer.pad_idx())
            piece_predictions = torch.argmax(pred_piece_flat, dim=-1)
            piece_accu = ((piece_predictions == gold_piece) & not_pad).sum().item() / not_pad.sum().item()

            # Edge prediction accuracy
            edge_predictions = torch.argmax(pred_edge_logits, dim=-1)
            edge_accu = (edge_predictions == golden_edge).sum().item() / len(golden_edge)

            return z_vecs, (loss_tuple, (piece_accu, edge_accu))

        return z_vecs, loss_tuple

    def inference(self, z, max_atom_num, add_edge_th, temperature):
        """
        Generate a molecule from a latent vector using the trained VAE decoder.

        This method implements the complete molecular generation pipeline:
        1. Sequential piece generation using RNN
        2. Molecular assembly from pieces
        3. Edge prediction and bond formation
        4. Chemical validation and cleanup

        Mathematical Background:
        - Uses temperature-controlled sampling: p(token) = softmax(logits / temperature)
        - Lower temperature → more deterministic generation
        - Higher temperature → more diverse generation
        - Edge confidence thresholding prevents spurious bond formation

        Args:
            z (Tensor): Latent vector [latent_dim]
            max_atom_num (int): Maximum number of atoms to generate
            add_edge_th (float): Confidence threshold for adding edges (0.0-1.0)
            temperature (float): Sampling temperature for piece generation

        Returns:
            Chem.Mol: Generated RDKit molecule object (sanitized and kekulized)
        """

        # =================================================================
        # STAGE 1: SETUP AND INITIALIZATION
        # =================================================================

        # Prepare latent vector for batch processing
        z = z.unsqueeze(0)  # [1, latent_dim] - add batch dimension
        batch_size = z.shape[0]  # Always 1 for inference

        # =================================================================
        # STAGE 2: SEQUENTIAL PIECE GENERATION
        # =================================================================

        # Initialize RNN with start token
        # Start with END token as initial input (common practice in sequence generation)
        cur_piece = self.piece_embedding(
            torch.tensor([[self.tokenizer.end_idx()]], dtype=torch.long, device=z.device)
        )  # [1, 1, embedding_dim]

        # Initialize RNN hidden state from latent vector
        # This conditions the entire sequence generation on the latent code
        hidden = self.latent_to_rnn_hidden(z).unsqueeze(0)  # [1, 1, hidden_dim]

        # Generation loop variables
        piece_ids = []  # Generated piece sequence
        cur_piece_id = None  # Current piece ID
        cur_atom_num = 0  # Running count of atoms

        # Generate pieces until stopping condition
        while cur_piece_id != self.tokenizer.end_idx():
            # =============================================================
            # RNN Forward Pass
            # =============================================================

            # Feed current piece embedding to RNN
            rnn_input = cur_piece  # [1, 1, embedding_dim]
            output, hidden = self.rnn(rnn_input, hidden)  # output: [1, 1, hidden_dim]

            # Convert RNN output to vocabulary logits
            output = self.to_vocab(output)  # [1, 1, num_piece_types]
            output = output.squeeze()  # [num_piece_types]

            # =============================================================
            # Sampling with Constraints
            # =============================================================

            # Mask invalid tokens
            output[self.tokenizer.pad_idx()] = float('-inf')  # Never generate padding

            # Force generation of at least one piece (prevent immediate termination)
            if len(piece_ids) == 0:
                output[self.tokenizer.end_idx()] = float('-inf')

            # Temperature-controlled sampling
            # Lower temperature → more deterministic, higher → more diverse
            probs = torch.softmax(output / temperature, dim=-1)  # [num_piece_types]
            cur_piece_id = torch.multinomial(probs, num_samples=1)  # Sample from distribution

            # Prepare next iteration
            cur_piece = self.piece_embedding(cur_piece_id).unsqueeze(0)  # [1, 1, embedding_dim]
            cur_piece_id = cur_piece_id.item()

            # =============================================================
            # Stopping Conditions
            # =============================================================

            # Count atoms in current piece
            cur_atom_num += cnt_atom(self.tokenizer.idx_to_piece(cur_piece_id))
            piece_ids.append(cur_piece_id)

            # Stop if atom limit exceeded
            if cur_atom_num > max_atom_num:
                break

            # Stop if position limit reached (sequence too long)
            if len(piece_ids) == self.pos_embedding.num_embeddings:
                break

        # Remove end token from generated sequence
        piece_ids = piece_ids[:-1]

        # =================================================================
        # STAGE 3: MOLECULAR ASSEMBLY FROM PIECES
        # =================================================================

        # Initialize molecular construction variables
        x = []  # Atom features (element types)
        edge_index = []  # Edge connectivity
        edge_attr = []  # Edge attributes (bond types)
        groups = []  # Atom groupings by piece
        aid2gid = {}  # Atom ID → Group ID mapping
        aid2bid = {}  # Atom ID → Block ID mapping (connected components)
        block_atom_cnt = []  # Atom count per connected block
        gen_mol = Chem.RWMol()  # RDKit molecule being constructed
        edge_sets = []  # Bond types connected to each atom
        x_pieces = []  # Piece ID for each atom
        x_pos = []  # Position ID for each atom

        # Process each generated piece
        for pos, pid in enumerate(piece_ids):
            # =============================================================
            # Convert Piece to Molecular Fragment
            # =============================================================

            # Get SMILES string and convert to molecule
            smi = self.tokenizer.idx_to_piece(pid)
            try:
                mol = smiles2molecule(smi, kekulize=True)
            except Exception:
                print(f"Failed to parse SMILES: {smi}")
                continue

            # =============================================================
            # Add Atoms from Current Piece
            # =============================================================

            offset = len(x)  # Current atom count (for indexing)
            group = []  # Atoms in current piece
            atom_num = mol.GetNumAtoms()

            for aid in range(atom_num):
                atom = mol.GetAtomWithIdx(aid)

                # Track atom in current group
                group.append(len(x))

                # Initialize mappings for new atom
                aid2gid[len(x)] = len(groups)  # Group assignment
                aid2bid[len(x)] = len(groups)  # Block assignment (initially separate)

                # Store atom features
                x.append(self.tokenizer.chem_vocab.atom_to_idx(atom.GetSymbol()))
                edge_sets.append([])  # Initialize bond list for this atom
                x_pieces.append(pid)  # Piece this atom came from
                x_pos.append(pos + 1)  # Position in sequence (1-indexed)

                # Add atom to RDKit molecule
                gen_mol.AddAtom(Chem.Atom(atom.GetSymbol()))

            groups.append(group)
            block_atom_cnt.append(atom_num)

            # =============================================================
            # Add Bonds within Current Piece
            # =============================================================

            for bond in mol.GetBonds():
                begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_type = self.tokenizer.chem_vocab.bond_to_idx(bond.GetBondType())

                # Adjust indices for global atom numbering
                begin_idx += offset
                end_idx += offset

                # Add bidirectional edges (undirected graph representation)
                edge_index.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
                edge_attr.extend([bond_type, bond_type])

                # Track bond types for each atom
                edge_sets[begin_idx].append(bond_type)
                edge_sets[end_idx].append(bond_type)

                # Add bond to RDKit molecule
                gen_mol.AddBond(begin_idx, end_idx, bond.GetBondType())

        # =================================================================
        # STAGE 4: GRAPH NEURAL NETWORK PROCESSING
        # =================================================================

        # Store original data for reference
        atoms = x

        # Create node embeddings (atom + piece + position information)
        node_x = self.embed_atom(
            torch.tensor(x, dtype=torch.long, device=z.device),
            torch.tensor(x_pieces, dtype=torch.long, device=z.device),
            torch.tensor(x_pos, dtype=torch.long, device=z.device)
        )  # [num_atoms, node_embedding_dim]

        # Prepare edge data for GNN
        if len(edge_index) == 0:
            # Handle case with no edges (single atoms only)
            edge_index = torch.randn(2, 0, device=z.device).long()
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=z.device).t().contiguous()

        # Convert edge attributes to one-hot encoding
        edge_attr = F.one_hot(
            torch.tensor(edge_attr, dtype=torch.long, device=z.device),
            num_classes=self.tokenizer.chem_vocab.num_bond_type()
        )

        # Apply graph neural network to get structural node embeddings
        node_embedding, _ = self.graph_embedding.embed_node(node_x, edge_index, edge_attr)
        # [num_atoms, node_embedding_dim]

        # =================================================================
        # STAGE 5: EDGE PREDICTION SETUP
        # =================================================================

        node_num = len(x)

        # Create edge selection mask (upper triangular to avoid duplicates)
        # Only predict edges between atoms from different pieces
        edge_select = torch.triu(torch.ones(node_num, node_num, dtype=torch.long, device=z.device))

        # Exclude intra-piece edges (already exist within molecular pieces)
        for group in groups:
            group_len = len(group)
            for i in range(group_len):
                for j in range(i, group_len):
                    m, n = group[i], group[j]
                    edge_select[m][n] = edge_select[n][m] = 0

        edge_select = edge_select.unsqueeze(0).bool()  # [1, node_num, node_num]

        # =================================================================
        # STAGE 6: INTER-PIECE EDGE PREDICTION
        # =================================================================

        # Prepare node embeddings for edge prediction
        node_embedding = node_embedding.unsqueeze(0)  # [1, num_atoms, embedding_dim]

        # Create source and target embeddings for all possible edges
        src_embedding = torch.repeat_interleave(
            node_embedding, node_num, dim=1
        ).view(batch_size, node_num, node_num, -1)
        src_embedding = src_embedding[edge_select]  # [num_potential_edges, embedding_dim]

        dst_embedding = torch.repeat_interleave(
            node_embedding, node_num, dim=0
        ).view(batch_size, node_num, node_num, -1)
        dst_embedding = dst_embedding[edge_select]  # [num_potential_edges, embedding_dim]

        # Add latent context to edge prediction
        latent_repeat = torch.repeat_interleave(
            z, node_num ** 2, dim=0
        ).view(batch_size, node_num, node_num, -1)
        latent_repeat = latent_repeat[edge_select]  # [num_potential_edges, latent_dim]

        # Combine features for edge prediction
        edge_pred_input = torch.cat([src_embedding, dst_embedding, latent_repeat], dim=-1)

        # =================================================================
        # STAGE 7: BOND FORMATION WITH CHEMICAL VALIDATION
        # =================================================================

        if edge_pred_input.shape[0] > 0:  # Check if there are edges to predict
            # Predict edge types and confidences
            pred_edge_logits = self.edge_predictor(edge_pred_input)  # [num_edges, num_edge_types]
            pred_edge_probs = torch.softmax(pred_edge_logits, dim=-1)

            # Get edge indices and predictions
            pred_edge_index = torch.nonzero(edge_select.squeeze())  # [num_edges, 2]
            none_bond = self.tokenizer.chem_vocab.bond_to_idx(None)
            confidence, edge_type = torch.max(pred_edge_probs, dim=-1)  # [num_edges]

            # Filter edges by confidence and bond type
            possible_edge_idx = [
                i for i in range(len(pred_edge_probs))
                if confidence[i] >= add_edge_th and edge_type[i] != none_bond
            ]

            # Sort by confidence (add most confident bonds first)
            sorted_idx = sorted(possible_edge_idx, key=lambda i: confidence[i], reverse=True)

            # Add bonds with chemical validation
            for i in sorted_idx:
                begin_idx, end_idx = pred_edge_index[i]
                begin_idx, end_idx = begin_idx.item(), end_idx.item()
                bond_type = edge_type[i]

                # Chemical validation checks
                valence_valid = valence_check(
                    atoms[begin_idx], atoms[end_idx],
                    edge_sets[begin_idx], edge_sets[end_idx],
                    bond_type, self.tokenizer.chem_vocab
                )

                # Cycle validation (only allow 5 or 6-membered rings)
                cycle_valid = cycle_check(begin_idx, end_idx, gen_mol)

                if valence_valid and cycle_valid:
                    # Add bond to molecule
                    bond_obj = self.tokenizer.chem_vocab.idx_to_bond(bond_type)
                    gen_mol.AddBond(begin_idx, end_idx, bond_obj)

                    # Update bond tracking
                    edge_sets[begin_idx].append(bond_type)
                    edge_sets[end_idx].append(bond_type)

                    # Update connected components
                    bid1, bid2 = aid2bid[begin_idx], aid2bid[end_idx]
                    if bid1 != bid2:
                        # Merge connected blocks
                        for aid in aid2bid:
                            if aid2bid[aid] == bid1:
                                aid2bid[aid] = bid2
                        block_atom_cnt[bid2] += block_atom_cnt[bid1]

        # =================================================================
        # STAGE 8: CLEANUP AND FINALIZATION
        # =================================================================

        # Remove disconnected fragments (keep largest connected component)
        # Find the block with maximum atom count
        bid = max(range(len(block_atom_cnt)), key=lambda i: block_atom_cnt[i])

        # Remove atoms not in the main connected component
        atoms_to_remove = sorted(
            [aid for aid in aid2bid.keys() if aid2bid[aid] != bid],
            reverse=True  # Remove from end to avoid index shifting
        )

        for aid in atoms_to_remove:
            gen_mol.RemoveAtom(aid)

        # =================================================================
        # STAGE 9: CHEMICAL SANITIZATION
        # =================================================================

        # Convert to final molecule object
        gen_mol = gen_mol.GetMol()

        # Sanitize molecule (check chemical validity)
        Chem.SanitizeMol(gen_mol)

        # Convert to Kekule form (explicit double bonds)
        Chem.Kekulize(gen_mol)

        return gen_mol

    def inference_constraint(self, z, max_atom_num, add_edge_th, temperature, constraint_mol):
        """
        Generate a molecule from a latent vector with structural constraints.

        This method extends the standard inference by starting with a constraint molecule
        and then generating additional pieces to expand the structure. The constraint
        molecule provides a fixed starting point that must be preserved in the output.

        Mathematical Background:
        - Constraint-based generation: z → pieces_constraint + pieces_new → molecule
        - RNN conditioning: Uses constraint pieces to initialize hidden state
        - Preserves original molecular scaffold while allowing expansion

        Args:
            z (Tensor): Latent vector [latent_dim]
            max_atom_num (int): Maximum number of atoms to generate
            add_edge_th (float): Confidence threshold for adding edges (0.0-1.0)
            temperature (float): Sampling temperature for piece generation
            constraint_mol (Chem.Mol): RDKit molecule object to use as constraint

        Returns:
            Chem.Mol: Generated RDKit molecule object (sanitized and converted to SMILES)
        """

        # =================================================================
        # STAGE 1: SETUP AND CONSTRAINT PROCESSING
        # =================================================================

        # Prepare latent vector for batch processing
        z = z.unsqueeze(0)  # [1, latent_dim] - add batch dimension
        batch_size = z.shape[0]  # Always 1 for inference

        # Preprocess constraint molecule
        # Ensure consistent representation (Kekule form for aromatic systems)
        Chem.Kekulize(constraint_mol)

        # Tokenize constraint molecule into pieces and atom groups
        # Returns: piece_ids (sequence of piece tokens), atom_groups (atoms per piece)
        init_pieces, init_groups = self.tokenizer.tokenize(constraint_mol, return_idx=True)

        # Initialize piece sequence with constraint pieces (excluding start/end tokens)
        piece_ids = list(init_pieces[1:-1])  # Remove <start> and <end> tokens
        cur_piece_id = None

        # Count atoms in constraint molecule
        cur_atom_num = 0
        for pid in piece_ids:
            cur_atom_num += cnt_atom(self.tokenizer.idx_to_piece(pid))

        # =================================================================
        # STAGE 2: RNN INITIALIZATION WITH CONSTRAINT CONTEXT
        # =================================================================

        # Initialize RNN with the last piece from constraint molecule
        # This conditions the generation on the existing structure
        cur_piece = self.piece_embedding(
            torch.tensor([[init_pieces[-2]]], dtype=torch.long, device=z.device)
        )  # [1, 1, embedding_dim] - last piece before <end> token

        # Prepare constraint sequence for RNN conditioning
        # Include all constraint pieces except the last one and <end> token
        init_rnn_input = torch.tensor(init_pieces[:-2], device=z.device)  # [seq_len]
        init_embeddings = self.piece_embedding(init_rnn_input).unsqueeze(0)  # [1, seq_len, embed_dim]

        # Clean up constraint data (remove start/end tokens)
        init_pieces = list(init_pieces[1:-1])  # Constraint piece IDs
        init_groups = list(init_groups[1:-1])  # Constraint atom groups
        init_mol = constraint_mol  # Store original constraint molecule

        # Initialize RNN hidden state from latent vector
        hidden = self.latent_to_rnn_hidden(z).unsqueeze(0)  # [1, 1, hidden_dim]

        # Process constraint sequence through RNN to condition hidden state
        # This "primes" the RNN with the constraint structure
        _, hidden = self.rnn(init_embeddings, hidden)  # Update hidden state with constraint context

        # =================================================================
        # STAGE 3: SEQUENTIAL PIECE GENERATION (EXPANSION)
        # =================================================================

        # Generate additional pieces to expand the constraint molecule
        while cur_piece_id != self.tokenizer.end_idx():
            # =============================================================
            # RNN Forward Pass
            # =============================================================

            # Feed current piece embedding to RNN
            rnn_input = cur_piece  # [1, 1, embedding_dim]
            output, hidden = self.rnn(rnn_input, hidden)  # output: [1, 1, hidden_dim]

            # Convert RNN output to vocabulary logits
            output = self.to_vocab(output)  # [1, 1, num_piece_types]
            output = output.squeeze()  # [num_piece_types]

            # =============================================================
            # Sampling with Constraints
            # =============================================================

            # Mask invalid tokens
            output[self.tokenizer.pad_idx()] = float('-inf')  # Never generate padding

            # Force generation of at least one new piece (prevent immediate termination)
            if len(piece_ids) == 0:
                output[self.tokenizer.end_idx()] = float('-inf')

            # Temperature-controlled sampling
            probs = torch.softmax(output / temperature, dim=-1)  # [num_piece_types]
            cur_piece_id = torch.multinomial(probs, num_samples=1)  # Sample from distribution

            # Prepare next iteration
            cur_piece = self.piece_embedding(cur_piece_id).unsqueeze(0)  # [1, 1, embedding_dim]
            cur_piece_id = cur_piece_id.item()

            # =============================================================
            # Stopping Conditions
            # =============================================================

            # Count atoms in current piece
            cur_atom_num += cnt_atom(self.tokenizer.idx_to_piece(cur_piece_id))
            piece_ids.append(cur_piece_id)

            # Stop if atom limit exceeded
            if cur_atom_num > max_atom_num:
                break

            # Stop if position limit reached (sequence too long)
            if len(piece_ids) == self.pos_embedding.num_embeddings:
                break

        # Remove end token from generated sequence
        piece_ids = piece_ids[:-1]

        # =================================================================
        # STAGE 4: MOLECULAR ASSEMBLY WITH CONSTRAINT PRESERVATION
        # =================================================================

        # Initialize molecular construction variables
        x = []  # Atom features (element types)
        edge_index = []  # Edge connectivity
        edge_attr = []  # Edge attributes (bond types)
        groups = []  # Atom groupings by piece
        aid2gid = {}  # Atom ID → Group ID mapping
        aid2bid = {}  # Atom ID → Block ID mapping (connected components)
        block_atom_cnt = []  # Atom count per connected block
        gen_mol = Chem.RWMol()  # RDKit molecule being constructed
        edge_sets = []  # Bond types connected to each atom
        x_pieces = []  # Piece ID for each atom
        x_pos = []  # Position ID for each atom

        # Track inter-group edges from constraint molecule
        inter_group_edges = []  # Edges between different constraint pieces
        init_aid2aid = {}  # Mapping from constraint atom IDs to new atom IDs

        # =================================================================
        # STAGE 5: PROCESS CONSTRAINT PIECES FIRST
        # =================================================================

        # Add atoms and bonds from constraint molecule
        for pos, pid in enumerate(piece_ids):
            # Only process constraint pieces (not newly generated ones)
            if pos == len(init_pieces):
                break

            # =============================================================
            # Add Atoms from Current Constraint Piece
            # =============================================================

            group = []  # Atoms in current piece
            atom_num = len(init_groups[pos])  # Number of atoms in this constraint piece

            for aid in init_groups[pos]:  # Iterate over constraint atom IDs
                atom = init_mol.GetAtomWithIdx(aid)

                # Track atom in current group
                group.append(len(x))
                init_aid2aid[aid] = len(x)  # Map constraint atom ID to new atom ID

                # Initialize mappings for new atom
                aid2gid[len(x)] = len(groups)  # Group assignment
                aid2bid[len(x)] = len(groups)  # Block assignment (initially separate)

                # Store atom features
                x.append(self.tokenizer.chem_vocab.atom_to_idx(atom.GetSymbol()))
                edge_sets.append([])  # Initialize bond list for this atom
                x_pieces.append(pid)  # Piece this atom came from
                x_pos.append(pos + 1)  # Position in sequence (1-indexed)

                # Add atom to RDKit molecule (preserve formal charge from constraint)
                atom_symbol = atom.GetSymbol()
                new_atom = Chem.Atom(atom_symbol)
                new_atom.SetFormalCharge(atom.GetFormalCharge())  # Preserve charge state
                gen_mol.AddAtom(new_atom)

            groups.append(group)
            block_atom_cnt.append(atom_num)

            # =============================================================
            # Add Bonds from Constraint Molecule
            # =============================================================

            for bond in init_mol.GetBonds():
                begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

                # Skip bonds involving atoms not in current constraint pieces
                if begin_idx not in init_aid2aid or end_idx not in init_aid2aid:
                    continue

                # Map to new atom indices
                begin_idx = init_aid2aid[begin_idx]
                end_idx = init_aid2aid[end_idx]

                # Skip bonds not involving current group
                if begin_idx not in group and end_idx not in group:
                    continue

                # Track inter-group edges (bonds between different pieces)
                if begin_idx not in group or end_idx not in group:
                    inter_group_edges.append((begin_idx, end_idx))

                # Add bond information
                bond_type = self.tokenizer.chem_vocab.bond_to_idx(bond.GetBondType())

                # Add bidirectional edges (undirected graph representation)
                edge_index.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
                edge_attr.extend([bond_type, bond_type])

                # Track bond types for each atom
                edge_sets[begin_idx].append(bond_type)
                edge_sets[end_idx].append(bond_type)

                # Add bond to RDKit molecule
                gen_mol.AddBond(begin_idx, end_idx, bond.GetBondType())

        # =================================================================
        # STAGE 6: PROCESS NEWLY GENERATED PIECES
        # =================================================================

        # Add atoms and bonds from newly generated pieces
        for pos, pid in enumerate(piece_ids):
            # Skip constraint pieces (already processed)
            if pos < len(init_pieces):
                continue

            # =============================================================
            # Convert Piece to Molecular Fragment
            # =============================================================

            # Get SMILES string and convert to molecule
            smi = self.tokenizer.idx_to_piece(pid)
            try:
                mol = smiles2molecule(smi, kekulize=True)
            except Exception:
                print(f"Failed to parse SMILES: {smi}")
                continue

            # =============================================================
            # Add Atoms from Current New Piece
            # =============================================================

            offset = len(x)  # Current atom count (for indexing)
            group = []  # Atoms in current piece
            atom_num = mol.GetNumAtoms()

            for aid in range(atom_num):
                atom = mol.GetAtomWithIdx(aid)

                # Track atom in current group
                group.append(len(x))

                # Initialize mappings for new atom
                aid2gid[len(x)] = len(groups)  # Group assignment
                aid2bid[len(x)] = len(groups)  # Block assignment (initially separate)

                # Store atom features
                x.append(self.tokenizer.chem_vocab.atom_to_idx(atom.GetSymbol()))
                edge_sets.append([])  # Initialize bond list for this atom
                x_pieces.append(pid)  # Piece this atom came from
                x_pos.append(pos + 1)  # Position in sequence (1-indexed)

                # Add atom to RDKit molecule
                gen_mol.AddAtom(Chem.Atom(atom.GetSymbol()))

            groups.append(group)
            block_atom_cnt.append(atom_num)

            # =============================================================
            # Add Bonds within Current New Piece
            # =============================================================

            for bond in mol.GetBonds():
                begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_type = self.tokenizer.chem_vocab.bond_to_idx(bond.GetBondType())

                # Adjust indices for global atom numbering
                begin_idx += offset
                end_idx += offset

                # Add bidirectional edges (undirected graph representation)
                edge_index.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
                edge_attr.extend([bond_type, bond_type])

                # Track bond types for each atom
                edge_sets[begin_idx].append(bond_type)
                edge_sets[end_idx].append(bond_type)

                # Add bond to RDKit molecule
                gen_mol.AddBond(begin_idx, end_idx, bond.GetBondType())

        # =================================================================
        # STAGE 7: GRAPH NEURAL NETWORK PROCESSING
        # =================================================================

        # Store original data for reference
        atoms = x

        # Create node embeddings (atom + piece + position information)
        node_x = self.embed_atom(
            torch.tensor(x, dtype=torch.long, device=z.device),
            torch.tensor(x_pieces, dtype=torch.long, device=z.device),
            torch.tensor(x_pos, dtype=torch.long, device=z.device)
        )  # [num_atoms, node_embedding_dim]

        # Prepare edge data for GNN
        if len(edge_index) == 0:
            # Handle case with no edges (single atoms only)
            edge_index = torch.randn(2, 0, device=z.device).long()
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=z.device).t().contiguous()

        # Convert edge attributes to one-hot encoding
        edge_attr = F.one_hot(
            torch.tensor(edge_attr, dtype=torch.long, device=z.device),
            num_classes=self.tokenizer.chem_vocab.num_bond_type()
        )

        # Apply graph neural network to get structural node embeddings
        node_embedding, _ = self.graph_embedding.embed_node(node_x, edge_index, edge_attr)
        # [num_atoms, node_embedding_dim]

        # =================================================================
        # STAGE 8: EDGE PREDICTION SETUP WITH CONSTRAINT AWARENESS
        # =================================================================

        node_num = len(x)

        # Create edge selection mask (upper triangular to avoid duplicates)
        # Only predict edges between atoms from different pieces
        edge_select = torch.triu(torch.ones(node_num, node_num, dtype=torch.long, device=z.device))

        # Exclude intra-piece edges (already exist within molecular pieces)
        for group in groups:
            group_len = len(group)
            for i in range(group_len):
                for j in range(i, group_len):
                    m, n = group[i], group[j]
                    edge_select[m][n] = edge_select[n][m] = 0

        # Exclude inter-group edges from constraint molecule (already exist)
        for begin_idx, end_idx in inter_group_edges:
            edge_select[begin_idx][end_idx] = edge_select[end_idx][begin_idx] = 0

            # Update connected components for constraint inter-group bonds
            bid1, bid2 = aid2bid[begin_idx], aid2bid[end_idx]
            if bid1 != bid2:
                # Merge connected blocks
                for aid in aid2bid:
                    if aid2bid[aid] == bid1:
                        aid2bid[aid] = bid2
                block_atom_cnt[bid2] += block_atom_cnt[bid1]

        edge_select = edge_select.unsqueeze(0).bool()  # [1, node_num, node_num]

        # =================================================================
        # STAGE 9: INTER-PIECE EDGE PREDICTION
        # =================================================================

        # Prepare node embeddings for edge prediction
        node_embedding = node_embedding.unsqueeze(0)  # [1, num_atoms, embedding_dim]

        # Create source and target embeddings for all possible edges
        src_embedding = torch.repeat_interleave(
            node_embedding, node_num, dim=1
        ).view(batch_size, node_num, node_num, -1)
        src_embedding = src_embedding[edge_select]  # [num_potential_edges, embedding_dim]

        dst_embedding = torch.repeat_interleave(
            node_embedding, node_num, dim=0
        ).view(batch_size, node_num, node_num, -1)
        dst_embedding = dst_embedding[edge_select]  # [num_potential_edges, embedding_dim]

        # Add latent context to edge prediction
        latent_repeat = torch.repeat_interleave(
            z, node_num ** 2, dim=0
        ).view(batch_size, node_num, node_num, -1)
        latent_repeat = latent_repeat[edge_select]  # [num_potential_edges, latent_dim]

        # Combine features for edge prediction
        edge_pred_input = torch.cat([src_embedding, dst_embedding, latent_repeat], dim=-1)

        # =================================================================
        # STAGE 10: BOND FORMATION WITH ENHANCED CHEMICAL VALIDATION
        # =================================================================

        if edge_pred_input.shape[0] > 0:  # Check if there are edges to predict
            # Predict edge types and confidences
            pred_edge_logits = self.edge_predictor(edge_pred_input)  # [num_edges, num_edge_types]
            pred_edge_probs = torch.softmax(pred_edge_logits, dim=-1)

            # Get edge indices and predictions
            pred_edge_index = torch.nonzero(edge_select.squeeze())  # [num_edges, 2]
            none_bond = self.tokenizer.chem_vocab.bond_to_idx(None)
            confidence, edge_type = torch.max(pred_edge_probs, dim=-1)  # [num_edges]

            # Filter edges by confidence and bond type
            possible_edge_idx = [
                i for i in range(len(pred_edge_probs))
                if confidence[i] >= add_edge_th and edge_type[i] != none_bond
            ]

            # Sort by confidence (add most confident bonds first)
            sorted_idx = sorted(possible_edge_idx, key=lambda i: confidence[i], reverse=True)

            # Add bonds with enhanced chemical validation
            for i in sorted_idx:
                begin_idx, end_idx = pred_edge_index[i]
                begin_idx, end_idx = begin_idx.item(), end_idx.item()
                bond_type = edge_type[i]

                # Enhanced chemical validation checks (includes formal charges)
                valence_valid = valence_check(
                    atoms[begin_idx], atoms[end_idx],
                    edge_sets[begin_idx], edge_sets[end_idx],
                    bond_type, self.tokenizer.chem_vocab,
                    gen_mol.GetAtomWithIdx(begin_idx).GetFormalCharge(),  # Consider formal charge
                    gen_mol.GetAtomWithIdx(end_idx).GetFormalCharge()  # Consider formal charge
                )

                # Cycle validation (only allow 5 or 6-membered rings)
                cycle_valid = cycle_check(begin_idx, end_idx, gen_mol)

                if valence_valid and cycle_valid:
                    # Add bond to molecule
                    bond_obj = self.tokenizer.chem_vocab.idx_to_bond(bond_type)
                    gen_mol.AddBond(begin_idx, end_idx, bond_obj)

                    # Update bond tracking
                    edge_sets[begin_idx].append(bond_type)
                    edge_sets[end_idx].append(bond_type)

                    # Update connected components
                    bid1, bid2 = aid2bid[begin_idx], aid2bid[end_idx]
                    if bid1 != bid2:
                        # Merge connected blocks
                        for aid in aid2bid:
                            if aid2bid[aid] == bid1:
                                aid2bid[aid] = bid2
                        block_atom_cnt[bid2] += block_atom_cnt[bid1]

        # =================================================================
        # STAGE 11: CONSTRAINT-AWARE CLEANUP
        # =================================================================

        # Remove disconnected fragments while preserving constraint molecule
        # Find the block containing the constraint molecule (must include atom 0)
        sorted_bids = sorted(range(len(block_atom_cnt)), key=lambda i: block_atom_cnt[i], reverse=True)

        # Ensure the constraint molecule is preserved
        for i in sorted_bids:
            if i == aid2bid[0]:  # Block containing the first constraint atom
                bid = i
                break

        # Remove atoms not in the main connected component (preserves constraint)
        atoms_to_remove = sorted(
            [aid for aid in aid2bid.keys() if aid2bid[aid] != bid],
            reverse=True  # Remove from end to avoid index shifting
        )

        for aid in atoms_to_remove:
            gen_mol.RemoveAtom(aid)

        # =================================================================
        # STAGE 12: ENHANCED CHEMICAL SANITIZATION
        # =================================================================

        # Convert to final molecule object
        gen_mol = gen_mol.GetMol()

        # Convert to SMILES and back for standardization
        # This ensures consistent representation and removes any artifacts
        smiles = Chem.MolToSmiles(gen_mol)
        gen_mol = Chem.MolFromSmiles(smiles)

        return gen_mol








