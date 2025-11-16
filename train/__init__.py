"""
NNUE Training Package for Chess Position Evaluation
"""

HAS_TORCH = True
TORCH_IMPORT_ERROR = None

try:
    from .model import NNUEModel, HalfKP, count_parameters
    from .config import TrainingConfig, get_config
    from .dataset import ChessPositionDataset, create_dataloaders, split_dataset
    from .train import train
    from .download_data import download_and_process_lichess_data
    from .upload_to_hf import upload_model_to_hf
except Exception as exc:  # pragma: no cover - handled in tests
    HAS_TORCH = False
    TORCH_IMPORT_ERROR = exc

    def _torch_missing(*_args, **_kwargs):
        raise RuntimeError(
            "PyTorch dependencies are unavailable; install torch to use train package"
        ) from exc

    NNUEModel = HalfKP = count_parameters = _torch_missing
    TrainingConfig = get_config = _torch_missing
    ChessPositionDataset = create_dataloaders = split_dataset = _torch_missing
    train = download_and_process_lichess_data = upload_model_to_hf = _torch_missing

__all__ = [
    'NNUEModel',
    'HalfKP',
    'count_parameters',
    'TrainingConfig',
    'get_config',
    'ChessPositionDataset',
    'create_dataloaders',
    'split_dataset',
    'train',
    'download_and_process_lichess_data',
    'upload_model_to_hf',
    'HAS_TORCH',
    'TORCH_IMPORT_ERROR',
]
