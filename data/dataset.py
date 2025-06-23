import kagglehub
import os


def load_dataset(dataset_name: str):
    DATASETS = {
        'tinyStories': {
            'path': 'danilkos0101/tinyStories-dataset',
            'train_file': "TinyStories_train_tokens.npy",
            'valid_file': "TinyStories_valid_tokens.npy"
        },
        'openWebText': {
            'path': 'danilkos0101/owt-dataset',
            'train_file': "Owt_train_tokens.npy",
            'valid_file': "Owt_valid_tokens.npy"
        }
    }

    if dataset_name not in DATASETS:
        available = ", ".join(f"'{name}'" for name in DATASETS.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. Available datasets: {available}"
        )

    dataset_info = DATASETS[dataset_name]
    path = kagglehub.dataset_download(dataset_info['path'])
    
    train_path = os.path.join(path, dataset_info['train_file'])
    valid_path = os.path.join(path, dataset_info['valid_file'])
    
    return train_path, valid_path



