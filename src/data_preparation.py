"""
Data preparation script with DVC tracking
"""
import os
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def download_and_prepare_data(config):
    """Download IMDB dataset and prepare for training"""
    logger.info("Starting data preparation...")
    
    # Create data directory
    data_dir = Path(config['paths']['data_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download IMDB dataset
    logger.info(f"Downloading {config['data']['dataset_name']} dataset...")
    dataset = load_dataset(config['data']['dataset_name'])
    
    # Convert to pandas for easier manipulation
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Use subset if specified
    if config['data']['max_samples']:
        logger.info(f"Using subset of {config['data']['max_samples']} samples")
        train_df = train_df.sample(n=min(config['data']['max_samples'], len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(config['data']['max_samples']//5, len(test_df)), random_state=42)
    
    # Split train into train and validation
    from sklearn.model_selection import train_test_split
    
    train_df, val_df = train_test_split(
        train_df,
        test_size=config['data']['val_size'] / (config['data']['train_size'] + config['data']['val_size']),
        random_state=config['training']['seed'],
        stratify=train_df['label']
    )
    
    # Save datasets
    logger.info("Saving datasets...")
    train_df.to_csv(data_dir / 'train.csv', index=False)
    val_df.to_csv(data_dir / 'val.csv', index=False)
    test_df.to_csv(data_dir / 'test.csv', index=False)
    
    # Print statistics
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    logger.info(f"\nTrain label distribution:\n{train_df['label'].value_counts()}")
    logger.info(f"\nValidation label distribution:\n{val_df['label'].value_counts()}")
    logger.info(f"\nTest label distribution:\n{test_df['label'].value_counts()}")
    
    # Save metadata
    metadata = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'total_size': len(train_df) + len(val_df) + len(test_df),
        'num_classes': train_df['label'].nunique(),
        'dataset_name': config['data']['dataset_name']
    }
    
    import json
    with open(data_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n✅ Data preparation complete!")
    logger.info(f"Files saved to {data_dir}")
    
    return metadata

if __name__ == "__main__":
    config = load_config()
    download_and_prepare_data(config)
