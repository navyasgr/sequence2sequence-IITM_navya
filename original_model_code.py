"""
Transliteration Model: Latin script to Devanagari script
Built for the Aksharantar dataset

This implements a character-level encoder-decoder architecture.
I've kept it modular so different cell types and layer configurations can be tested easily.

Author: Navya
Date: October 2025
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharEncoder(nn.Module):
    """
    Encodes a sequence of Latin characters into a fixed representation.
    
    The encoder reads each character one by one, maintaining a hidden state
    that accumulates information about the word. The final hidden state captures
    the essence of the entire input word.
    
    Architecture choices:
    - Supports RNN, LSTM, and GRU (LSTM works best in my tests)
    - Dropout between layers for regularization
    - Packing for efficiency with variable lengths
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1, 
                 cell_type='LSTM', dropout_prob=0.3):
        super(CharEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cell_type = cell_type
        
        # Character embedding layer - converts indices to dense vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0  # 0 is reserved for padding
        )
        
        # Initialize embeddings with small random values
        # (helps with gradient flow early in training)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Select RNN cell type
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout_prob if n_layers > 1 else 0
            )
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout_prob if n_layers > 1 else 0
            )
        elif cell_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout_prob if n_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown cell type: {cell_type}. Use 'RNN', 'LSTM', or 'GRU'")
        
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_seq, seq_lengths):
        """
        Process a batch of input sequences.
        
        Args:
            input_seq: (batch_size, max_seq_len) - character indices
            seq_lengths: (batch_size,) - actual length of each sequence
            
        Returns:
            outputs: (batch_size, max_seq_len, hidden_dim) - all hidden states
            final_hidden: final hidden state(s) to pass to decoder
        """
        batch_size = input_seq.size(0)
        
        # Convert character indices to embeddings
        embedded = self.embedding(input_seq)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        
        # Pack sequences to ignore padding during computation
        # This is crucial for efficiency and correct gradient flow
        packed_input = pack_padded_sequence(
            embedded, 
            seq_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Process through RNN
        if self.cell_type == 'LSTM':
            packed_output, (hidden, cell_state) = self.rnn(packed_input)
            final_hidden = (hidden, cell_state)
        else:
            packed_output, hidden = self.rnn(packed_input)
            final_hidden = hidden
        
        # Unpack to get back padded sequences
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        return outputs, final_hidden


class CharDecoder(nn.Module):
    """
    Decodes the encoded representation into Devanagari characters.
    
    The decoder generates output one character at a time. At each step:
    1. Takes the previous character (or <SOS> at start)
    2. Runs it through the RNN with current hidden state
    3. Projects to vocabulary size to get probabilities for next character
    
    Key feature: Uses the encoder's final hidden state to initialize,
    giving it context about what word to generate.
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1,
                 cell_type='LSTM', dropout_prob=0.3):
        super(CharDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.cell_type = cell_type
        
        # Output character embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # RNN cell (must match encoder type for state compatibility)
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout_prob if n_layers > 1 else 0
            )
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout_prob if n_layers > 1 else 0
            )
        elif cell_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout_prob if n_layers > 1 else 0
            )
        
        # Project hidden state to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward_step(self, current_char, hidden_state):
        """
        Process one decoding step.
        
        Args:
            current_char: (batch_size, 1) - current character index
            hidden_state: hidden state from previous step
            
        Returns:
            output: (batch_size, vocab_size) - scores for each character
            hidden_state: updated hidden state
        """
        # Embed current character
        embedded = self.embedding(current_char)  # (batch, 1, embed_dim)
        embedded = self.dropout(embedded)
        
        # Run through RNN for one step
        if self.cell_type == 'LSTM':
            rnn_out, (hidden, cell) = self.rnn(embedded, hidden_state)
            hidden_state = (hidden, cell)
        else:
            rnn_out, hidden = self.rnn(embedded, hidden_state)
            hidden_state = hidden
        
        # Project to vocabulary
        output = self.output_projection(rnn_out.squeeze(1))  # (batch, vocab_size)
        
        return output, hidden_state
    
    def forward(self, encoder_hidden, target_seq, teacher_forcing_ratio=0.5):
        """
        Full decoding for training (uses teacher forcing).
        
        Args:
            encoder_hidden: final hidden state from encoder
            target_seq: (batch_size, max_len) - target sequence (for teacher forcing)
            teacher_forcing_ratio: probability of using true target vs model's prediction
            
        Returns:
            outputs: (batch_size, max_len, vocab_size) - scores for each position
        """
        batch_size = target_seq.size(0)
        max_len = target_seq.size(1)
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(target_seq.device)
        
        # Start with encoder's hidden state
        hidden_state = encoder_hidden
        
        # First input is <SOS> token (index 1)
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long).to(target_seq.device)
        
        # Generate sequence step by step
        for t in range(max_len):
            output, hidden_state = self.forward_step(decoder_input, hidden_state)
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing and t < max_len - 1:
                # Use actual target as next input
                decoder_input = target_seq[:, t].unsqueeze(1)
            else:
                # Use model's own prediction
                decoder_input = output.argmax(dim=1).unsqueeze(1)
        
        return outputs


class Seq2SeqTransliterator(nn.Module):
    """
    Complete sequence-to-sequence transliteration model.
    
    Combines encoder and decoder to convert Latin script to Devanagari.
    The encoder builds a representation of the input word, and the decoder
    uses that representation to generate the output word character by character.
    
    This architecture is flexible - you can easily swap RNN types, adjust sizes,
    or add more layers without changing the core logic.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=128, 
                 hidden_dim=256, n_layers=1, cell_type='LSTM', dropout=0.3):
        super(Seq2SeqTransliterator, self).__init__()
        
        self.encoder = CharEncoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            cell_type=cell_type,
            dropout_prob=dropout
        )
        
        self.decoder = CharDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            cell_type=cell_type,
            dropout_prob=dropout
        )
        
        self.cell_type = cell_type
    
    def forward(self, src_seq, src_lengths, tgt_seq, teacher_forcing_ratio=0.5):
        """
        Forward pass for training.
        
        Args:
            src_seq: (batch, src_len) - source sequences
            src_lengths: (batch,) - actual lengths
            tgt_seq: (batch, tgt_len) - target sequences
            teacher_forcing_ratio: how often to use true targets
            
        Returns:
            outputs: (batch, tgt_len, tgt_vocab_size) - predictions
        """
        # Encode source sequence
        _, encoder_hidden = self.encoder(src_seq, src_lengths)
        
        # Decode to target sequence
        outputs = self.decoder(encoder_hidden, tgt_seq, teacher_forcing_ratio)
        
        return outputs
    
    def translate(self, src_seq, src_lengths, max_length=50):
        """
        Inference mode - generate translations without teacher forcing.
        
        Args:
            src_seq: (batch, src_len) - source sequences
            src_lengths: (batch,) - actual lengths
            max_length: maximum output length
            
        Returns:
            predictions: (batch, pred_len) - predicted character indices
        """
        batch_size = src_seq.size(0)
        device = src_seq.device
        
        # Encode
        _, encoder_hidden = self.encoder(src_seq, src_lengths)
        
        # Initialize decoder
        hidden_state = encoder_hidden
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long).to(device)
        
        # Store predictions
        predictions = []
        
        # Generate until <EOS> or max_length
        for _ in range(max_length):
            output, hidden_state = self.decoder.forward_step(decoder_input, hidden_state)
            
            # Get most likely character
            predicted_char = output.argmax(dim=1)
            predictions.append(predicted_char.unsqueeze(1))
            
            # Check if all sequences have generated <EOS> (index 2)
            if (predicted_char == 2).all():
                break
            
            # Use prediction as next input
            decoder_input = predicted_char.unsqueeze(1)
        
        # Concatenate all predictions
        predictions = torch.cat(predictions, dim=1)
        
        return predictions
    
    def count_parameters(self):
        """Count trainable parameters - useful for understanding model size."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_theoretical_complexity(embed_dim, hidden_dim, seq_len, vocab_size, cell_type='LSTM'):
    """
    Calculate theoretical computational complexity and parameter count.
    
    This is what I calculated for the assignment requirements.
    
    Args:
        embed_dim (m): embedding dimension
        hidden_dim (h): hidden state dimension
        seq_len (n): average sequence length
        vocab_size (V): vocabulary size
        cell_type: 'RNN', 'LSTM', or 'GRU'
    
    Returns:
        dict with computations and parameters
    """
    m, h, n, V = embed_dim, hidden_dim, seq_len, vocab_size
    
    if cell_type == 'LSTM':
        # LSTM has 4 gates, each needs input and hidden transformations
        encoder_ops = n * 4 * (h*m + h*h)
        decoder_ops = n * 4 * (h*m + h*h)
        
        # Each gate has: input weights, hidden weights, and biases
        encoder_params = 4 * (h*m + h*h + h)
        decoder_params = 4 * (h*m + h*h + h)
        
    elif cell_type == 'GRU':
        # GRU has 3 gates
        encoder_ops = n * 3 * (h*m + h*h)
        decoder_ops = n * 3 * (h*m + h*h)
        encoder_params = 3 * (h*m + h*h + h)
        decoder_params = 3 * (h*m + h*h + h)
        
    else:  # Vanilla RNN
        encoder_ops = n * (h*m + h*h)
        decoder_ops = n * (h*m + h*h)
        encoder_params = h*m + h*h + h
        decoder_params = h*m + h*h + h
    
    # Output projection and embeddings are same for all types
    output_ops = n * h * V
    embedding_params = V * m
    output_params = h * V + V
    
    total_ops = encoder_ops + decoder_ops + output_ops
    total_params = embedding_params + encoder_params + decoder_params + output_params
    
    return {
        'total_operations': total_ops,
        'encoder_operations': encoder_ops,
        'decoder_operations': decoder_ops,
        'output_operations': output_ops,
        'total_parameters': total_params,
        'embedding_parameters': embedding_params,
        'encoder_parameters': encoder_params,
        'decoder_parameters': decoder_params,
        'output_parameters': output_params,
    }


if __name__ == "__main__":
    # Quick test to verify model works
    print("Testing model architecture...")
    
    # Small test configuration
    src_vocab = 50
    tgt_vocab = 60
    
    model = Seq2SeqTransliterator(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        embed_dim=32,
        hidden_dim=64,
        n_layers=1,
        cell_type='LSTM'
    )
    
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Test with dummy batch
    batch_size = 4
    src_len = 10
    tgt_len = 12
    
    dummy_src = torch.randint(0, src_vocab, (batch_size, src_len))
    dummy_tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))
    dummy_lengths = torch.tensor([10, 8, 9, 7])
    
    # Training mode
    output = model(dummy_src, dummy_lengths, dummy_tgt, teacher_forcing_ratio=0.5)
    print(f"Training output shape: {output.shape}")
    
    # Inference mode
    predictions = model.translate(dummy_src, dummy_lengths, max_length=15)
    print(f"Inference output shape: {predictions.shape}")
    
    # Theoretical analysis
    print("\nTheoretical Complexity:")
    complexity = compute_theoretical_complexity(128, 256, 15, 100, 'LSTM')
    print(f"Total operations per sequence: {complexity['total_operations']:,}")
    print(f"Total parameters: {complexity['total_parameters']:,}")
    
    print("\nModel test passed!")