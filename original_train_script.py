"""
Training script for the transliteration model.

I've structured this to handle both GPU and CPU training, with proper
checkpointing and monitoring. The script saves the best model based on
validation performance and logs everything for later analysis.

Usage:
    python train.py --config config.yaml
    python train.py --epochs 50 --batch_size 32 --device cuda
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
import os
from pathlib import Path
import json

from transliteration_model import Seq2SeqTransliterator, compute_theoretical_complexity
from data_utils import load_dataset, TransliterationDataset, build_vocabularies
import config as cfg


def train_one_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio, clip_grad):
    """
    Train for one complete epoch.
    
    Returns average loss for the epoch.
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (src, tgt, src_lens, tgt_lens) in enumerate(dataloader):
        # Move to device
        src = src.to(device)
        tgt = tgt.to(device)
        src_lens = src_lens.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(src, src_lens, tgt, teacher_forcing_ratio)
        
        # Compute loss (ignore padding and first SOS token)
        # predictions shape: (batch, seq_len, vocab_size)
        # tgt shape: (batch, seq_len)
        
        # Reshape for loss computation
        pred_flat = predictions[:, :-1, :].contiguous().view(-1, predictions.size(-1))
        tgt_flat = tgt[:, 1:].contiguous().view(-1)
        
        loss = criterion(pred_flat, tgt_flat)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent explosion
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set.
    
    Returns average loss.
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            src_lens = src_lens.to(device)
            
            # Forward pass without teacher forcing
            predictions = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
            
            # Compute loss
            pred_flat = predictions[:, :-1, :].contiguous().view(-1, predictions.size(-1))
            tgt_flat = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(pred_flat, tgt_flat)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def compute_word_accuracy(model, dataloader, src_vocab, tgt_vocab, device, num_samples=None):
    """
    Calculate word-level accuracy (exact match).
    
    This is stricter than character accuracy - the entire word must be correct.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for src, tgt, src_lens, _ in dataloader:
            src = src.to(device)
            src_lens = src_lens.to(device)
            
            # Generate predictions
            predictions = model.translate(src, src_lens)
            
            batch_size = src.size(0)
            for i in range(batch_size):
                # Convert to lists, removing special tokens
                pred_chars = predictions[i].cpu().tolist()
                tgt_chars = tgt[i].cpu().tolist()
                
                # Clean up (remove padding, SOS, EOS)
                pred_clean = [c for c in pred_chars if c not in [0, 1, 2]]
                tgt_clean = [c for c in tgt_chars if c not in [0, 1, 2]]
                
                # Check exact match
                if pred_clean == tgt_clean:
                    correct += 1
                total += 1
                
                # Early stop if sampling
                if num_samples and total >= num_samples:
                    break
            
            if num_samples and total >= num_samples:
                break
    
    return correct / total if total > 0 else 0.0


def show_predictions(model, dataloader, src_vocab, tgt_vocab, device, num_examples=5):
    """
    Display some example predictions to see how the model is doing.
    """
    model.eval()
    shown = 0
    
    print("\n" + "="*80)
    print("Sample Predictions:")
    print("="*80)
    
    with torch.no_grad():
        for src, tgt, src_lens, _ in dataloader:
            src = src.to(device)
            src_lens = src_lens.to(device)
            
            predictions = model.translate(src, src_lens)
            
            for i in range(min(src.size(0), num_examples - shown)):
                # Convert to characters
                src_chars = [src_vocab.idx2char[idx] for idx in src[i].cpu().tolist() 
                            if idx not in [0, 1, 2]]
                tgt_chars = [tgt_vocab.idx2char[idx] for idx in tgt[i].cpu().tolist() 
                            if idx not in [0, 1, 2]]
                pred_chars = [tgt_vocab.idx2char.get(idx, '?') for idx in predictions[i].cpu().tolist() 
                             if idx not in [0, 1, 2]]
                
                src_word = ''.join(src_chars)
                tgt_word = ''.join(tgt_chars)
                pred_word = ''.join(pred_chars)
                
                match = "✓" if pred_word == tgt_word else "✗"
                print(f"{match} '{src_word}' -> '{tgt_word}' (predicted: '{pred_word}')")
                
                shown += 1
                if shown >= num_examples:
                    break
            
            if shown >= num_examples:
                break
    
    print("="*80 + "\n")


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, accuracy, filepath):
    """Save model checkpoint with all relevant information."""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'accuracy': accuracy,
        'config': {
            'embed_dim': cfg.EMBED_DIM,
            'hidden_dim': cfg.HIDDEN_DIM,
            'n_layers': cfg.NUM_LAYERS,
            'cell_type': cfg.CELL_TYPE,
        }
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load a saved checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


def main(args):
    """Main training pipeline."""
    
    print("\n" + "="*80)
    print("Transliteration Model Training")
    print("="*80)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
    
    # Load data
    print("\nLoading dataset...")
    train_pairs, val_pairs, test_pairs = load_dataset(
        args.data_path, 
        train_ratio=0.8, 
        val_ratio=0.1
    )
    
    print(f"Training samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    print(f"Test samples: {len(test_pairs)}")
    
    # Build vocabularies
    print("\nBuilding vocabularies...")
    src_vocab, tgt_vocab = build_vocabularies(train_pairs, min_freq=cfg.MIN_FREQ)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Create datasets
    train_dataset = TransliterationDataset(train_pairs, src_vocab, tgt_vocab)
    val_dataset = TransliterationDataset(val_pairs, src_vocab, tgt_vocab)
    test_dataset = TransliterationDataset(test_pairs, src_vocab, tgt_vocab)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=0
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = Seq2SeqTransliterator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=cfg.EMBED_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        n_layers=cfg.NUM_LAYERS,
        cell_type=cfg.CELL_TYPE,
        dropout=cfg.DROPOUT
    ).to(device)
    
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    
    # Print theoretical complexity
    print("\n" + "-"*80)
    print("Theoretical Analysis:")
    print("-"*80)
    complexity = compute_theoretical_complexity(
        cfg.EMBED_DIM, cfg.HIDDEN_DIM, 15, len(tgt_vocab), cfg.CELL_TYPE
    )
    print(f"Operations per sequence: ~{complexity['total_operations']:,}")
    print(f"Theoretical parameters: {complexity['total_parameters']:,}")
    print(f"Actual parameters: {num_params:,}")
    print("-"*80)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            cfg.TEACHER_FORCING_RATIO, cfg.GRAD_CLIP
        )
        
        # Validate
        val_loss = evaluate_model(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 0.0, save_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Show sample predictions every few epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            show_predictions(model, val_loader, src_vocab, tgt_vocab, device, num_examples=5)
        
        print()
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80 + "\n")
    
    # Load best model
    best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_checkpoint_path):
        _, _, best_val = load_checkpoint(best_checkpoint_path, model)
        print(f"Loaded best model (val_loss: {best_val:.4f})")
    
    # Test accuracy
    test_loss = evaluate_model(model, test_loader, criterion, device)
    test_accuracy = compute_word_accuracy(model, test_loader, src_vocab, tgt_vocab, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Show test predictions
    show_predictions(model, test_loader, src_vocab, tgt_vocab, device, num_examples=10)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'config': vars(args)
    }
    
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! History saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train transliteration model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/aksharantar_hindi.csv',
                       help='Path to dataset file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    main(args)