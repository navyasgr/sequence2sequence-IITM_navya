"""
Data utilities for loading and processing the Aksharantar dataset.

This handles all the data preparation:
- Loading CSV files
- Building vocabularies from scratch
- Creating PyTorch datasets
- Batching with proper padding

The key challenge is handling variable-length sequences efficiently.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from collections import Counter
import config as cfg


class Vocabulary:
    """
    Manages character-to-index mapping for a language.
    
    This is crucial for converting text to numbers the model can process.
    We build this from training data only (not test data) to avoid leakage.
    """
    
    def __init__(self, name='vocab'):
        self.name = name
        
        # Initialize with special tokens
        self.char2idx = {
            '<PAD>': cfg.PAD_IDX,
            '<SOS>': cfg.SOS_IDX,
            '<EOS>': cfg.EOS_IDX,
            '<UNK>': cfg.UNK_IDX
        }
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        
        # Counter for character frequencies
        self.char_freq = Counter()
        
        # Start after special tokens
        self.n_chars = 4
    
    def add_word(self, word):
        """Add all characters from a word to frequency counter."""
        for char in word:
            self.char_freq[char] += 1
    
    def build(self, min_freq=1):
        """
        Build the vocabulary from counted characters.
        
        Only includes characters that appear at least min_freq times.
        This filters out typos and extremely rare characters.
        """
        for char, freq in self.char_freq.items():
            if freq >= min_freq and char not in self.char2idx:
                self.char2idx[char] = self.n_chars
                self.idx2char[self.n_chars] = char
                self.n_chars += 1
        
        print(f"{self.name} vocabulary built: {self.n_chars} characters")
        print(f"  (filtered out {sum(1 for f in self.char_freq.values() if f < min_freq)} rare characters)")
    
    def encode(self, word):
        """
        Convert word to list of indices.
        
        Unknown characters get mapped to <UNK> token.
        """
        return [self.char2idx.get(char, cfg.UNK_IDX) for char in word]
    
    def decode(self, indices):
        """Convert list of indices back to word."""
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices])
    
    def __len__(self):
        return self.n_chars


def load_dataset(filepath, train_ratio=0.8, val_ratio=0.1):
    """
    Load Aksharantar dataset and split into train/val/test.
    
    Expected CSV format:
    latin_word,devanagari_word
    ghar,घर
    ajanabee,अजनबी
    ...
    
    Returns three lists of (source, target) tuples.
    """
    print(f"Loading data from {filepath}...")
    
    try:
        # Try reading with header
        df = pd.read_csv(filepath)
    except:
        # Try without header
        df = pd.read_csv(filepath, header=None, names=['source', 'target'])
    
    # Get word pairs
    pairs = list(zip(df.iloc[:, 0].values, df.iloc[:, 1].values))
    
    # Remove any NaN or empty pairs
    pairs = [(s, t) for s, t in pairs if isinstance(s, str) and isinstance(t, str) 
             and len(s) > 0 and len(t) > 0]
    
    print(f"Loaded {len(pairs)} word pairs")
    
    # Shuffle
    np.random.shuffle(pairs)
    
    # Split
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    
    return train_pairs, val_pairs, test_pairs


def build_vocabularies(train_pairs, min_freq=2):
    """
    Build source and target vocabularies from training data only.
    
    This is important - we don't look at validation or test data
    when building vocabularies to avoid data leakage.
    """
    print("\nBuilding vocabularies...")
    
    src_vocab = Vocabulary(name='Source (Latin)')
    tgt_vocab = Vocabulary(name='Target (Devanagari)')
    
    # Count character frequencies
    for src_word, tgt_word in train_pairs:
        src_vocab.add_word(src_word)
        tgt_vocab.add_word(tgt_word)
    
    # Build vocabularies (filtering by min_freq)
    src_vocab.build(min_freq)
    tgt_vocab.build(min_freq)
    
    # Show some stats
    print(f"\nMost common source characters:")
    for char, freq in src_vocab.char_freq.most_common(10):
        print(f"  '{char}': {freq}")
    
    print(f"\nMost common target characters:")
    for char, freq in tgt_vocab.char_freq.most_common(10):
        print(f"  '{char}': {freq}")
    
    return src_vocab, tgt_vocab


class TransliterationDataset(Dataset):
    """
    PyTorch dataset for transliteration pairs.
    
    Handles conversion of text to indices and proper sequence formatting.
    """
    
    def __init__(self, pairs, src_vocab, tgt_vocab):
        """
        Args:
            pairs: list of (source_word, target_word) tuples
            src_vocab: source language vocabulary
            tgt_vocab: target language vocabulary
        """
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get one example.
        
        Returns:
            src_tensor: source sequence with EOS
            tgt_tensor: target sequence with SOS and EOS
        """
        src_word, tgt_word = self.pairs[idx]
        
        # Encode to indices
        src_indices = self.src_vocab.encode(src_word)
        tgt_indices = self.tgt_vocab.encode(tgt_word)
        
        # Add special tokens
        # Source: word + EOS
        src_indices.append(cfg.EOS_IDX)
        
        # Target: SOS + word + EOS
        # SOS is added by decoder, we just add EOS here
        tgt_indices.append(cfg.EOS_IDX)
        
        # Convert to tensors
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
        
        return src_tensor, tgt_tensor
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for batching.
        
        Handles variable-length sequences by:
        1. Padding all sequences to max length in batch
        2. Tracking original lengths for pack_padded_sequence
        
        This is more efficient than padding everything to a fixed max length.
        """
        # Separate sources and targets
        src_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]
        
        # Get original lengths (before padding)
        src_lengths = torch.tensor([len(s) for s in src_batch])
        tgt_lengths = torch.tensor([len(t) for t in tgt_batch])
        
        # Pad sequences to max length in this batch
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=cfg.PAD_IDX)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=cfg.PAD_IDX)
        
        return src_padded, tgt_padded, src_lengths, tgt_lengths


def create_sample_data(num_samples=1000):
    """
    Create sample data for testing when actual dataset isn't available.
    
    Generates synthetic Latin-Devanagari pairs.
    Obviously not as good as real data, but useful for debugging.
    """
    print("Generating synthetic data for testing...")
    
    # Some common mappings
    mappings = [
        ('namaste', 'नमस्ते'),
        ('bharat', 'भारत'),
        ('ghar', 'घर'),
        ('paani', 'पानी'),
        ('kitab', 'किताब'),
        ('dost', 'दोस्त'),
        ('raat', 'रात'),
        ('din', 'दिन'),
        ('aasman', 'आसमान'),
        ('dharti', 'धरती'),
        ('suraj', 'सूरज'),
        ('chaand', 'चाँद'),
        ('pyaar', 'प्यार'),
        ('khushi', 'खुशी'),
        ('sapna', 'सपना'),
        ('zindagi', 'ज़िंदगी'),
        ('dil', 'दिल'),
        ('mann', 'मन'),
        ('baat', 'बात'),
        ('kaam', 'काम')
    ]
    
    # Repeat to get desired number
    pairs = []
    while len(pairs) < num_samples:
        pairs.extend(mappings)
    
    return pairs[:num_samples]


def analyze_dataset(pairs):
    """
    Print statistics about the dataset.
    
    Useful for understanding data distribution and potential issues.
    """
    print("\n" + "="*80)
    print("Dataset Analysis")
    print("="*80)
    
    src_lengths = [len(s) for s, t in pairs]
    tgt_lengths = [len(t) for s, t in pairs]
    
    print(f"\nSource (Latin) sequences:")
    print(f"  Average length: {np.mean(src_lengths):.2f}")
    print(f"  Min length: {np.min(src_lengths)}")
    print(f"  Max length: {np.max(src_lengths)}")
    print(f"  Median length: {np.median(src_lengths):.2f}")
    
    print(f"\nTarget (Devanagari) sequences:")
    print(f"  Average length: {np.mean(tgt_lengths):.2f}")
    print(f"  Min length: {np.min(tgt_lengths)}")
    print(f"  Max length: {np.max(tgt_lengths)}")
    print(f"  Median length: {np.median(tgt_lengths):.2f}")
    
    # Check for potential issues
    very_long = [(s, t) for s, t in pairs if len(s) > 30 or len(t) > 30]
    if very_long:
        print(f"\n⚠️ Warning: {len(very_long)} pairs have very long sequences (>30 chars)")
        print("  This might cause memory issues. Consider filtering or truncating.")
    
    # Character set sizes
    src_chars = set(''.join(s for s, t in pairs))
    tgt_chars = set(''.join(t for s, t in pairs))
    
    print(f"\nUnique characters:")
    print(f"  Source: {len(src_chars)}")
    print(f"  Target: {len(tgt_chars)}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Quick test of data loading
    print("Testing data utilities...\n")
    
    # Try to load real data, fall back to synthetic
    try:
        train, val, test = load_dataset('data/aksharantar_hindi.csv')
    except:
        print("Real data not found, using synthetic data")
        all_pairs = create_sample_data(1000)
        n = len(all_pairs)
        train = all_pairs[:int(n*0.8)]
        val = all_pairs[int(n*0.8):int(n*0.9)]
        test = all_pairs[int(n*0.9):]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")
    
    # Analyze
    analyze_dataset(train)
    
    # Build vocabularies
    src_vocab, tgt_vocab = build_vocabularies(train, min_freq=2)
    
    # Create dataset
    dataset = TransliterationDataset(train[:10], src_vocab, tgt_vocab)
    
    print(f"\nDataset created with {len(dataset)} examples")
    print("\nFirst example:")
    src, tgt = dataset[0]
    print(f"  Source indices: {src.tolist()}")
    print(f"  Target indices: {tgt.tolist()}")
    print(f"  Source word: {src_vocab.decode([i for i in src.tolist() if i not in [0,1,2]])}")
    print(f"  Target word: {tgt_vocab.decode([i for i in tgt.tolist() if i not in [0,1,2]])}")
    
    print("\nData utilities test passed!")