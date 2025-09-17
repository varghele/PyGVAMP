import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from psevo.models.psvae import PSVAEModel
from psevo.pipe.trainer import PSVAETrainer
from psevo.tokenizer.mol_bpe_old import Tokenizer
from psevo.dataset.bpe_dataset import BPEMolDataset, get_dataloader



def parse():
    """parse command"""
    parser = argparse.ArgumentParser(description='training overall model for molecule generation')
    parser.add_argument('--train_set', type=str, required=True, help='path of training dataset')
    parser.add_argument('--valid_set', type=str, required=True, help='path of validation dataset')
    parser.add_argument('--test_set', type=str, required=True, help='path of test dataset')
    parser.add_argument('--vocab', type=str, required=True, help='path of vocabulary (.pkl) or bpe vocab(.txt)')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--save_dir', type=str, required=True, help='path to store the model')
    parser.add_argument('--batch_size', type=int, default=32, help='size of mini-batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--alpha', type=float, required=True,
                        help='balancing reconstruct loss and predictor loss')
    # vae training
    parser.add_argument('--beta', type=float, default=0,
                        help='balancing kl loss and other loss')
    parser.add_argument('--step_beta', type=float, default=0.002,
                        help='value of beta increasing step')
    parser.add_argument('--max_beta', type=float, default=0.01)
    parser.add_argument('--kl_warmup', type=int, default=0,
                        help='Within these steps beta is set to 0')
    parser.add_argument('--kl_anneal_iter', type=int, default=1000)

    parser.add_argument('--num_workers', type=int, default=4, help='number of cpus to load data')
    parser.add_argument('--gpus', default=None, help='gpus to use')
    parser.add_argument('--epochs', type=int, default=6, help='max epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping number of epochs')
    parser.add_argument('--grad_clip', type=float, default=10.0,
                        help='clip large gradient to prevent gradient boom')
    parser.add_argument('--monitor', type=str, default='val_loss',
                        help='Value to monitor in early stopping')

    # model parameters
    parser.add_argument('--props', type=str, nargs='+', choices=['qed', 'sa', 'logp', 'gsk3b', 'jnk3'],
                        default=['qed', 'logp'], help='properties to predict')
    parser.add_argument('--predictor_hidden_dim', type=int, default=200,
                        help='hidden dim of predictor (MLP)')
    parser.add_argument('--node_hidden_dim', type=int, default=300,
                        help='dim of node hidden embedding in encoder and decoder')
    parser.add_argument('--graph_embedding_dim', type=int, default=400,
                        help='dim of graph embedding by encoder and also condition for ae decoders')
    parser.add_argument('--latent_dim', type=int, default=56,
                        help='dim of latent z for vae decoders')
    # ps-vae decoder only
    parser.add_argument('--max_pos', type=int, default=50,
                        help='Max number of pieces')
    parser.add_argument('--atom_embedding_dim', type=int, default=50,
                        help='Embedding dim for a single atom')
    parser.add_argument('--piece_embedding_dim', type=int, default=100,
                        help='Embedding dim for piece')
    parser.add_argument('--pos_embedding_dim', type=int, default=50,
                        help='Position embedding of piece')
    parser.add_argument('--piece_hidden_dim', type=int, default=200,
                        help='Hidden dim for rnn used in piece generation')
    return parser.parse_args()


def get_default_args():
    """
    Generate default arguments for PS-VAE training with fixed dataset paths.

    This function provides a convenient way to set up training arguments
    without requiring command line input. Useful for notebooks, scripts,
    or when you want to programmatically set training parameters.

    Returns:
        argparse.Namespace: Namespace object with default training arguments
    """
    import argparse

    # Create namespace object to mimic parsed arguments
    args = argparse.Namespace()

    # Fixed dataset paths
    base_pth = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(base_pth)
    args.train_set = os.path.join(base_pth, 'data/zinc250k/train.txt')
    args.valid_set = os.path.join(base_pth, 'data/zinc250k/valid.txt')
    args.test_set = os.path.join(base_pth, 'data/zinc250k/test.txt')
    args.vocab = os.path.join(base_pth, 'area52/vocab_zinc250')
    args.save_dir = 'checkpoints/'
    args.device = 'cuda'

    # Training parameters
    args.shuffle = True
    args.batch_size = 256
    args.lr = 1e-3
    args.epochs = 10
    args.patience = 3
    args.grad_clip = 10.0
    args.num_workers = 4

    # Loss weighting parameters
    args.alpha = 0.7  # Balance between reconstruction and prediction
    args.beta = 0.0  # Base KL weight
    args.step_beta = 0.002
    args.max_beta = 0.01
    args.kl_warmup = 1000
    args.kl_anneal_iter = 1000

    # Model architecture parameters
    args.encoder_type = 'cheb'  # or 'cheb', 'ml3'
    args.props = ['qed', 'logp']
    args.predictor_hidden_dim = 200
    args.node_hidden_dim = 300
    args.graph_embedding_dim = 400
    args.latent_dim = 56

    # PS-VAE decoder specific parameters
    args.max_pos = 50
    args.atom_embedding_dim = 50
    args.piece_embedding_dim = 100
    args.pos_embedding_dim = 50
    args.piece_hidden_dim = 200

    return args


def build_config(args, tokenizer):
    """Build model configuration from arguments."""
    # Map property names to indices
    property_map = {'qed': 0, 'sa': 1, 'logp': 2, 'gsk3b': 3, 'jnk3': 4}
    selected_properties = [property_map[prop] for prop in args.props]

    # Calculate input dimension for encoder (sum of embedding dimensions)
    encoder_input_dim = args.atom_embedding_dim + args.piece_embedding_dim + args.pos_embedding_dim

    config = {
        # Training parameters
        'lr': args.lr,
        'alpha': args.alpha,
        'beta': args.beta,
        'step_beta': args.step_beta,
        'max_beta': args.max_beta,
        'kl_warmup': args.kl_warmup,
        'kl_anneal_iter': args.kl_anneal_iter,
        'grad_clip': args.grad_clip,
        'patience': args.patience,
        'selected_properties': selected_properties,

        # Encoder configuration
        'encoder': {
            'type': args.encoder_type,
            'dim_in': encoder_input_dim,
            'dim_hidden': args.node_hidden_dim,
            'dim_out': args.graph_embedding_dim,
            'num_edge_type': 5,  # Standard bond types
            't': 4  # Number of message passing iterations
        },

        # Predictor configuration
        'predictor': {
            'input_dim': args.latent_dim,
            'hidden_dim': args.predictor_hidden_dim,
            'output_dim': len(selected_properties)
        },

        # VAE piece decoder configuration
        'vae_piece_decoder': {
            'atom_embedding_dim': args.atom_embedding_dim,
            'piece_embedding_dim': args.piece_embedding_dim,
            'max_pos': args.max_pos,
            'pos_embedding_dim': args.pos_embedding_dim,
            'piece_hidden_dim': args.piece_hidden_dim,
            'node_hidden_dim': args.node_hidden_dim,
            'num_edge_type': 5,
            'cond_dim': args.graph_embedding_dim,
            'latent_dim': args.latent_dim,
            'tokenizer': tokenizer,
            't': 4
        }
    }

    return config


# Configuration
config = {
    'lr': 1e-3,
    'alpha': 0.7,  # Balance between reconstruction and prediction
    'beta': 0.0,   # Base KL weight
    'step_beta': 0.002,  # KL annealing step
    'max_beta': 0.01,    # Maximum KL weight
    'kl_warmup': 1000,   # KL warmup steps
    'kl_anneal_iter': 1000,  # KL annealing iterations
    'grad_clip': 10.0,   # Gradient clipping
    'selected_properties': [0, 1, 2],  # Which properties to predict
    'encoder': {
        'type': 'ml3',  # 'gin' or 'cheb', 'ml3'
        'dim_in': 200, # TODO: This is atom+piece+pos embedding
        'dim_hidden': 128,
        'dim_out': 256,
        'num_edge_type': 4,
        't': 4
    },
    'predictor': {
        'input_dim': 56,
        'hidden_dim': 200,
        'output_dim': 3
    },
    'vae_piece_decoder': {
        'atom_embedding_dim': 50,
        'piece_embedding_dim': 100,
        'max_pos': 50,
        'pos_embedding_dim': 50,
        'piece_hidden_dim': 200,
        'node_hidden_dim': 300,
        'num_edge_type': 4,
        'cond_dim': 256,
        'latent_dim': 56,
        't': 4
    }
}

def main():
    """Main training function."""
    args = get_default_args()
    #args = parse()

    print("Loading tokenizer...")
    tokenizer = Tokenizer(args.vocab)

    print("Building model configuration...")
    #config = build_config(args, tokenizer)

    print("Creating data loaders...")
    train_loader = get_dataloader(
        args.train_set, tokenizer,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers
    )

    val_loader = get_dataloader(
        args.valid_set, tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_loader = get_dataloader(
        args.test_set, tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print("Initializing model...")
    model = PSVAEModel(config, tokenizer).to(args.device)

    print("Creating trainer...")
    trainer = PSVAETrainer(model, config, device='cuda' if torch.cuda.is_available() else 'cpu')

    print("Starting training...")
    print(f"Model: PS-VAE with {config['encoder']['type'].upper()} encoder")
    print(f"Properties: {args.props}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Alpha (rec/pred balance): {args.alpha}")

    # Train the model
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )

    print("Training completed!")

    # Optional: Test the model
    print("Running final test...")
    test_metrics = []
    for batch in test_loader:
        metrics = trainer.model.validation_step(batch.to(args.device), trainer.global_step)
        test_metrics.append(metrics)

    # Average test metrics
    avg_test_metrics = {}
    for key in test_metrics[0].keys():
        if key != 'encoder_type':
            avg_test_metrics[key] = sum(m[key] for m in test_metrics) / len(test_metrics)

    print("Test Results:")
    print(f"  Test Loss: {avg_test_metrics['val_total_loss']:.4f}")
    print(f"  Piece Accuracy: {avg_test_metrics['val_piece_accuracy']:.4f}")
    print(f"  Edge Accuracy: {avg_test_metrics['val_edge_accuracy']:.4f}")


if __name__ == '__main__':
    main()
    #args = parse()

    #tokenizer = Tokenizer(args.vocab)

    # Initialize model and trainer
    #model = PSVAEModel(config, tokenizer)
    #trainer = PSVAETrainer(model, config, device='cuda')

    # Train the model
    #trainer.fit(train_loader, val_loader, num_epochs=100, save_dir='./checkpoints')

    #print("PS-VAE model and trainer ready for use!")
