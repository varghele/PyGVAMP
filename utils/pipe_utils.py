import os
import torch
import json
import datetime
from vampnet import VAMPNet

def save_run_info(model, train_config, dataset_info, save_dir, run_id=None):
    """
    Save comprehensive information about a training run.

    Parameters
    ----------
    model : VAMPNet
        Trained model
    train_config : dict
        Training configuration parameters
    dataset_info : dict
        Information about the dataset used
    save_dir : str
        Directory to save information
    run_id : str, optional
        Unique identifier for this run

    Returns
    -------
    str
        Path to the saved run directory
    """
    # Create a unique run ID if not provided
    if run_id is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"

    run_dir = os.path.join(save_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(run_dir, "model.pt")
    model.save(model_path, metadata={
        'train_config': train_config,
        'dataset_info': dataset_info
    })

    # Save configuration as separate JSON for easy access
    model_config = model.get_config()
    with open(os.path.join(run_dir, "model_config.json"), 'w') as f:
        json.dump(model_config, f, indent=2)

    with open(os.path.join(run_dir, "train_config.json"), 'w') as f:
        json.dump(train_config, f, indent=2)

    with open(os.path.join(run_dir, "dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Run information saved to {run_dir}")
    return run_dir


def load_run(run_dir, encoder_class, vamp_score_class,
             embedding_class=None, classifier_class=None):
    """
    Load a complete training run.

    Parameters
    ----------
    run_dir : str
        Path to the run directory
    encoder_class : class
        Class of the encoder network
    vamp_score_class : class
        Class of the VAMP score module
    embedding_class : class, optional
        Class of the embedding module
    classifier_class : class, optional
        Class of the classifier module

    Returns
    -------
    tuple
        (loaded_model, train_config, dataset_info)
    """
    model_path = os.path.join(run_dir, "model.pt")

    # Load model and metadata
    model, metadata = VAMPNet.load(
        model_path,
        encoder_class=encoder_class,
        vamp_score_class=vamp_score_class,
        embedding_class=embedding_class,
        classifier_class=classifier_class
    )

    # Load configurations
    train_config = metadata.get('train_config', {})
    dataset_info = metadata.get('dataset_info', {})

    # Try loading from separate files if not in metadata
    if not train_config:
        try:
            with open(os.path.join(run_dir, "train_config.json"), 'r') as f:
                train_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Warning: Could not load training configuration")

    if not dataset_info:
        try:
            with open(os.path.join(run_dir, "dataset_info.json"), 'r') as f:
                dataset_info = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Warning: Could not load dataset information")

    return model, train_config, dataset_info
