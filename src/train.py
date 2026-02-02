"""
Training script with W&B experiment tracking
"""
import os
import wandb
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="configs/config.yaml"):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class WandbCallback:
    """Custom callback for additional W&B logging"""
    def __init__(self):
        self.training_loss = []
        self.eval_loss = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                self.training_loss.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_loss.append(logs['eval_loss'])

def train_model(config):
    """Main training function"""
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Initialize W&B
    logger.info("Initializing Weights & Biases...")
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb'].get('entity'),
        config=config,
        name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=['distilbert', 'sentiment-analysis', 'imdb']
    )
    
    # Load data
    logger.info("Loading datasets...")
    data_dir = Path(config['paths']['data_dir'])
    
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config['model']['max_length']
        )
    
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Load model
    logger.info(f"Loading model: {config['model']['name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'],
        num_labels=config['model']['num_labels']
    )
    
    # Training arguments
    output_dir = Path(config['paths']['model_dir']) / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="steps",
        eval_steps=config['training']['eval_steps'],
        save_strategy="steps",
        save_steps=config['training']['save_steps'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        num_train_epochs=config['training']['epochs'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=config['training']['save_total_limit'],
        report_to="wandb",
        seed=config['training']['seed'],
        fp16=torch.cuda.is_available(),
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60 + "\n")
    
    train_result = trainer.train()
    
    # Evaluate on validation set
    logger.info("\n" + "="*60)
    logger.info("EVALUATING ON VALIDATION SET")
    logger.info("="*60 + "\n")
    
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    logger.info(f"Validation Results: {val_results}")
    
    # Evaluate on test set
    logger.info("\n" + "="*60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*60 + "\n")
    
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    logger.info(f"Test Results: {test_results}")
    
    # Get predictions for confusion matrix
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Log confusion matrix to W&B
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels,
            preds=preds,
            class_names=['Negative', 'Positive']
        )
    })
    
    # Save model
    logger.info("\n" + "="*60)
    logger.info("SAVING MODEL")
    logger.info("="*60 + "\n")
    
    model_dir = Path(config['paths']['model_dir']) / 'sentiment_model'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Save metrics
    metrics = {
        'model_name': config['model']['name'],
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'val_accuracy': val_results.get('eval_accuracy'),
        'val_f1': val_results.get('eval_f1'),
        'test_accuracy': test_results.get('eval_accuracy'),
        'test_f1': test_results.get('eval_f1'),
        'test_precision': test_results.get('eval_precision'),
        'test_recall': test_results.get('eval_recall'),
        'confusion_matrix': cm.tolist(),
        'training_time': train_result.metrics.get('train_runtime'),
        'total_epochs': config['training']['epochs'],
        'training_date': datetime.now().isoformat()
    }
    
    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save config used for training
    with open(model_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"Model saved to {model_dir}")
    logger.info(f"\n=== Final Metrics ===")
    logger.info(f"Test Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
    logger.info(f"Test F1 Score: {test_results.get('eval_f1', 0):.4f}")
    
    # Log model to W&B
    if config['wandb']['log_model']:
        logger.info("\nLogging model to W&B...")
        wandb.save(str(model_dir / '*'))
    
    wandb.finish()
    
    return metrics

if __name__ == "__main__":
    config = load_config()
    metrics = train_model(config)
