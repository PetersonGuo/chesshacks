"""
Module for uploading NNUE model to Hugging Face Hub
"""

import os
import json
import torch
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from huggingface_hub import HfApi, upload_folder, login, get_token

from .config import TrainingConfig, get_config
from .model import HalfKP

def create_model_card(
    model_name: str,
    model_description: str = "",
    training_config: Optional[Dict[str, Any]] = None,
    checkpoint_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a model card (README.md) for the Hugging Face model

    Args:
        model_name: Name of the model
        model_description: Description of the model
        training_config: Training configuration dictionary
        checkpoint_info: Checkpoint information (epoch, losses, etc.)

    Returns:
        Model card content as string
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract checkpoint info
    epoch = checkpoint_info.get('epoch', 'N/A') if checkpoint_info else 'N/A'
    train_loss = checkpoint_info.get('train_loss', 'N/A') if checkpoint_info else 'N/A'
    val_loss = checkpoint_info.get('val_loss', 'N/A') if checkpoint_info else 'N/A'

    # Extract model architecture from training config
    hidden_size = training_config.get('hidden_size', 256) if training_config else 256
    hidden2_size = training_config.get('hidden2_size', 32) if training_config else 32
    hidden3_size = training_config.get('hidden3_size', 32) if training_config else 32

    # Format losses
    train_loss_str = f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else str(train_loss)
    val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)

    card = f"""# {model_name}

{model_description if model_description else "NNUE (Efficiently Updatable Neural Network) chess evaluation model"}

## Model Details

- **Model type**: NNUE Chess Evaluation
- **Architecture**: HalfKP feature representation
- **Uploaded**: {current_time}

## Architecture

```
Input: HalfKP features (40,960 dimensions per perspective)
  ↓
Feature Transformer: 40,960 → {hidden_size} (separate for white/black)
  ↓
ClippedReLU activation
  ↓
Concatenate: {hidden_size} + {hidden_size} → {hidden_size * 2}
  ↓
Hidden Layer 1: {hidden_size * 2} → {hidden2_size} + ClippedReLU
  ↓
Hidden Layer 2: {hidden2_size} → {hidden3_size} + ClippedReLU
  ↓
Output Layer: {hidden3_size} → 1 (centipawn evaluation)
```

## Training Information

- **Epoch**: {epoch + 1 if isinstance(epoch, int) else epoch}
- **Training Loss**: {train_loss_str}
- **Validation Loss**: {val_loss_str}

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
import chess

# Download and load model
checkpoint_path = hf_hub_download(repo_id="{model_name}", filename="pytorch_model.bin")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Load model config
model_config = checkpoint['model_config']

# Create model instance (you'll need the NNUEModel class)
# from model import NNUEModel
# model = NNUEModel(**model_config)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# Evaluate a position
# board = chess.Board()
# score = model.evaluate_board(board)
# print(f"Evaluation: {{score:.2f}} centipawns")
```

## Training Configuration

{_format_training_config(training_config) if training_config else "No training configuration available"}

---

*Model generated with NNUE training pipeline*
"""

    return card


def _format_training_config(config: Dict[str, Any]) -> str:
    """Format training configuration for model card"""
    if not config:
        return ""

    # Filter out None values and format nicely
    formatted = []
    key_params = [
        'batch_size', 'learning_rate', 'num_epochs', 'optimizer',
        'loss_function', 'hidden_size', 'hidden2_size', 'hidden3_size'
    ]

    for key in key_params:
        if key in config and config[key] is not None:
            value = config[key]
            formatted.append(f"- **{key}**: {value}")

    return "\n".join(formatted) if formatted else "No configuration details available"


def load_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load checkpoint information without loading the full model
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return {
        'epoch': checkpoint.get('epoch', None),
        'train_loss': checkpoint.get('train_loss', None),
        'val_loss': checkpoint.get('val_loss', None),
        'config': checkpoint.get('config', None),
        'training_start_time': checkpoint.get('training_start_time', None),
    }


def prepare_model_for_upload(
    checkpoint_path: str,
    output_dir: str,
    model_config: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
    """
    Prepare model files for upload to Hugging Face
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save prepared files
        model_config: Model architecture config (hidden_size, hidden2_size, hidden3_size)
    
    Returns:
        Tuple of (checkpoint_info, model_config, training_config)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model config from checkpoint if not provided
    if model_config is None:
        checkpoint_config = checkpoint.get('config', {})
        model_config = {
            'hidden_size': checkpoint_config.get('hidden_size', 256),
            'hidden2_size': checkpoint_config.get('hidden2_size', 32),
            'hidden3_size': checkpoint_config.get('hidden3_size', 32),
        }
    
    # Save model state dict
    model_state_dict = checkpoint['model_state_dict']
    torch.save({
        'model_state_dict': model_state_dict,
        'model_config': model_config,
    }, os.path.join(output_dir, 'pytorch_model.bin'))
    
    # Save model config as JSON
    config_dict = {
        'model_type': 'nnue',
        'architecture': {
            'hidden_size': model_config['hidden_size'],
            'hidden2_size': model_config['hidden2_size'],
            'hidden3_size': model_config['hidden3_size'],
            'feature_size': HalfKP.FEATURE_SIZE,
        },
        'training_config': checkpoint.get('config', {}),
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Return checkpoint info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', None),
        'train_loss': checkpoint.get('train_loss', None),
        'val_loss': checkpoint.get('val_loss', None),
        'training_start_time': checkpoint.get('training_start_time', None),
    }
    
    training_config = checkpoint.get('config', {})
    
    return checkpoint_info, model_config, training_config


def upload_model_to_hf(
    checkpoint_path: str,
    repo_id: Optional[str] = None,
    model_name: Optional[str] = None,
    model_description: Optional[str] = None,
    private: Optional[bool] = None,
    token: Optional[str] = None,
    model_config: Optional[Dict[str, int]] = None,
    upload_all_checkpoints: Optional[bool] = None,
    checkpoints_dir: Optional[str] = None,
    config: Optional[TrainingConfig] = None,
    training_start_time: Optional[str] = None,
) -> str:
    """
    Upload model to Hugging Face Hub
    
    Args:
        checkpoint_path: Path to model checkpoint file
        repo_id: Hugging Face repository ID (defaults to config.hf_repo_id)
        model_name: Display name for the model (defaults to config.hf_model_name or repo_id)
        model_description: Description for the model card (defaults to config.hf_model_description)
        private: Whether the repository should be private (defaults to config.hf_private)
        token: Hugging Face token (if not provided, will try to use saved token)
        model_config: Model architecture config (hidden_size, hidden2_size, hidden3_size)
        upload_all_checkpoints: If True, upload all checkpoint files (defaults to config.hf_upload_all_checkpoints)
        checkpoints_dir: Directory containing checkpoints (defaults to config.hf_checkpoints_dir or checkpoint directory)
        config: TrainingConfig instance (if None, uses default config)
        training_start_time: ISO format timestamp when training started (will be extracted from checkpoint if not provided)
    
    Returns:
        URL of the uploaded model
    """
    # Load config if not provided
    if config is None:
        config = get_config('default')
    
    # Use config defaults if parameters not provided
    if repo_id is None:
        repo_id = config.hf_repo_id
    if model_name is None:
        model_name = config.hf_model_name
    if model_description is None:
        model_description = config.hf_model_description
    if private is None:
        private = config.hf_private
    if upload_all_checkpoints is None:
        upload_all_checkpoints = config.hf_upload_all_checkpoints
    if checkpoints_dir is None:
        checkpoints_dir = config.hf_checkpoints_dir
    # Check authentication
    if token:
        login(token=token)
    elif not get_token():
        print("No Hugging Face token found. Please login:")
        print("  huggingface-cli auth login")
        print("  or pass token parameter")
        raise ValueError("Hugging Face authentication required")
    
    # Create temporary directory for upload files
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare model files
        print(f"Preparing model files from {checkpoint_path}...")
        checkpoint_info, model_config, training_config = prepare_model_for_upload(
            checkpoint_path, temp_dir, model_config
        )
        
        # Use provided training_start_time if available, otherwise try to get from checkpoint
        if training_start_time is None:
            training_start_time = checkpoint_info.get('training_start_time')
        
        # Copy checkpoint file(s) to upload directory
        checkpoint_filename = os.path.basename(checkpoint_path)
        checkpoint_dest = os.path.join(temp_dir, checkpoint_filename)
        shutil.copy2(checkpoint_path, checkpoint_dest)
        print(f"Added checkpoint file: {checkpoint_filename}")
        
        # Optionally upload all checkpoints from the checkpoints directory
        if upload_all_checkpoints:
            if checkpoints_dir is None:
                checkpoints_dir = os.path.dirname(checkpoint_path) or config.checkpoint_dir
            
            if os.path.isdir(checkpoints_dir):
                print(f"Uploading all checkpoints from {checkpoints_dir}...")
                for filename in os.listdir(checkpoints_dir):
                    if filename.endswith('.pt'):
                        src_path = os.path.join(checkpoints_dir, filename)
                        # Skip if already added
                        if filename != checkpoint_filename:
                            dest_path = os.path.join(temp_dir, filename)
                            shutil.copy2(src_path, dest_path)
                            print(f"  Added checkpoint: {filename}")
            else:
                print(f"Warning: Checkpoints directory not found: {checkpoints_dir}")
        
        # Create model card
        card_content = create_model_card(
            model_name=model_name or repo_id.split('/')[-1],
            model_description=model_description,
            training_config=training_config,
            checkpoint_info=checkpoint_info,
        )
        
        with open(os.path.join(temp_dir, 'README.md'), 'w') as f:
            f.write(card_content)
        
        # Check if repository exists, throw error if it doesn't
        api = HfApi()
        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
        except Exception as e:
            raise ValueError(
                f"Repository '{repo_id}' does not exist. Please create it first on Hugging Face Hub.\n"
                f"Error: {e}"
            )
        
        # Create commit message with epoch and training start time
        epoch = checkpoint_info.get('epoch', 'N/A')
        epoch_display = epoch + 1 if isinstance(epoch, int) else epoch
        
        if training_start_time:
            try:
                # Parse ISO format and format nicely
                from datetime import datetime
                start_dt = datetime.fromisoformat(training_start_time.replace('Z', '+00:00'))
                start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                start_time_str = training_start_time
            commit_message = f"Epoch {epoch_display} - Training started: {start_time_str}"
        else:
            commit_message = f"Epoch {epoch_display}"
        
        # Upload files
        print(f"Uploading to {repo_id}...")
        upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        
        model_url = f"https://huggingface.co/{repo_id}"
        print(f"✓ Model uploaded successfully!")
        print(f"  URL: {model_url}")
        
        return model_url


def download_model_from_hf(
    repo_id: str,
    local_dir: str,
    token: Optional[str] = None,
) -> str:
    """
    Download model from Hugging Face Hub
    
    Args:
        repo_id: Hugging Face repository ID
        local_dir: Local directory to save the model
        token: Hugging Face token (optional)
    
    Returns:
        Path to downloaded checkpoint file
    """
    from huggingface_hub import snapshot_download
    
    print(f"Downloading model from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token,
    )
    
    checkpoint_path = os.path.join(local_dir, 'pytorch_model.bin')
    if os.path.exists(checkpoint_path):
        print(f"✓ Model downloaded to {checkpoint_path}")
        return checkpoint_path
    else:
        raise FileNotFoundError(f"Model file not found at {checkpoint_path}")


def main():
    """CLI interface for uploading models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload NNUE model to Hugging Face')
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        'repo_id',
        type=str,
        nargs='?',
        default=None,
        help='Hugging Face repository ID (defaults to config.hf_repo_id)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Display name for the model'
    )
    parser.add_argument(
        '--description',
        type=str,
        default=None,
        help='Model description'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the repository private'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face token (or set HF_TOKEN env var)'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=None,
        help='Model hidden size (if not in checkpoint)'
    )
    parser.add_argument(
        '--hidden2-size',
        type=int,
        default=None,
        help='Model hidden2 size (if not in checkpoint)'
    )
    parser.add_argument(
        '--hidden3-size',
        type=int,
        default=None,
        help='Model hidden3 size (if not in checkpoint)'
    )
    parser.add_argument(
        '--upload-all-checkpoints',
        action='store_true',
        help='Upload all checkpoint files from the checkpoints directory'
    )
    parser.add_argument(
        '--checkpoints-dir',
        type=str,
        default=None,
        help='Directory containing checkpoints (defaults to directory of checkpoint file)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = get_config('default')
    
    model_config = None
    if args.hidden_size or args.hidden2_size or args.hidden3_size:
        model_config = {
            'hidden_size': args.hidden_size or 256,
            'hidden2_size': args.hidden2_size or 32,
            'hidden3_size': args.hidden3_size or 32,
        }
    
    token = args.token or os.getenv('HF_TOKEN')
    
    upload_model_to_hf(
        checkpoint_path=args.checkpoint,
        repo_id=args.repo_id,
        model_name=args.name,
        model_description=args.description,
        private=args.private if args.private else None,
        token=token,
        model_config=model_config,
        upload_all_checkpoints=args.upload_all_checkpoints if args.upload_all_checkpoints else None,
        checkpoints_dir=args.checkpoints_dir,
        config=config,
    )


if __name__ == '__main__':
    main()
