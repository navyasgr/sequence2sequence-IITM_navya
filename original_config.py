"""
Configuration file for transliteration model.

All hyperparameters in one place. This makes it easy to:
- Try different architectures
- Tune hyperparameters
- Share exact settings with others

I've set these based on my experiments, but they're all adjustable.
"""

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# Character embedding dimension (m in the math)
# Larger = more expressive but more parameters
# I found 128 works well without being too heavy
EMBED_DIM = 128

# Hidden state dimension (h in the math)
# This is the "memory" capacity of the network
# 256 was the sweet spot for my GPU limitations
HIDDEN_DIM = 256

# Number of stacked RNN layers
# More layers = more capacity but slower training
# 1 layer works surprisingly well for this task
NUM_LAYERS = 1

# RNN cell type: 'RNN', 'LSTM', or 'GRU'
# LSTM handles long-term dependencies best
# GRU is faster but slightly less accurate
# vanilla RNN is fastest but struggles with long words
CELL_TYPE = 'LSTM'

# Dropout probability
# Helps prevent overfitting
# 0.3 means randomly drop 30% of connections during training
DROPOUT = 0.3

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Batch size
# Larger = faster training but more memory
# Had to use 32 due to GPU limitations (tried 64 and 128 but got OOM errors)
BATCH_SIZE = 32

# Learning rate
# How big of steps the optimizer takes
# 0.001 is a safe starting point
# Will be reduced automatically if validation loss plateaus
LEARNING_RATE = 0.001

# Number of training epochs
# One epoch = one pass through entire dataset
# 50 epochs usually enough for convergence
NUM_EPOCHS = 50

# Teacher forcing ratio
# Probability of feeding true target vs model's prediction during training
# 0.5 = 50% of time use true target, 50% use model's own prediction
# This helps model learn faster while avoiding over-reliance
TEACHER_FORCING_RATIO = 0.5

# Gradient clipping threshold
# Prevents exploding gradients
# If gradient norm exceeds this, it gets scaled down
GRAD_CLIP = 5.0

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Minimum character frequency
# Characters appearing less than this are treated as <UNK>
# Helps reduce vocab size and filter typos
MIN_FREQ = 2

# Maximum sequence length
# Sequences longer than this get truncated
# Most Hindi words are under 20 characters
MAX_LENGTH = 50

# Data split ratios
TRAIN_RATIO = 0.8  # 80% for training
VAL_RATIO = 0.1    # 10% for validation
TEST_RATIO = 0.1   # 10% for testing (calculated as remainder)

# ============================================================================
# SPECIAL TOKENS
# ============================================================================

# These indices are reserved for special purposes
PAD_IDX = 0   # Padding token (for batching variable lengths)
SOS_IDX = 1   # Start of sequence (decoder's first input)
EOS_IDX = 2   # End of sequence (signals completion)
UNK_IDX = 3   # Unknown character (for rare or unseen chars)

# Token names (for display)
SPECIAL_TOKENS = {
    PAD_IDX: '<PAD>',
    SOS_IDX: '<SOS>',
    EOS_IDX: '<EOS>',
    UNK_IDX: '<UNK>'
}

# ============================================================================
# PATHS
# ============================================================================

# Data paths
DATA_DIR = 'data'
TRAIN_FILE = 'aksharantar_hindi.csv'

# Model save paths
CHECKPOINT_DIR = 'checkpoints'
BEST_MODEL_PATH = 'checkpoints/best_model.pth'

# Log paths
LOG_DIR = 'logs'
TENSORBOARD_DIR = 'runs'

# ============================================================================
# HARDWARE NOTES
# ============================================================================

"""
My hardware limitations and what worked:

GPU: NVIDIA MX250 (2GB VRAM)
- Can't fit batches larger than 32
- Hidden dim above 256 causes OOM
- Had to reduce dataset size for initial experiments

Recommendations for better hardware:
- BATCH_SIZE = 128 (with 8GB+ VRAM)
- HIDDEN_DIM = 512 (with 8GB+ VRAM)
- NUM_LAYERS = 2 (with 8GB+ VRAM)

For CPU only:
- Set BATCH_SIZE = 16 (for reasonable speed)
- Training will take ~10x longer
- But accuracy should be similar
"""

# ============================================================================
# THEORETICAL COMPLEXITY REFERENCE
# ============================================================================

"""
For LSTM with 1 layer each:

Computations per sequence:
- Encoder: n × 4(h×m + h²) where n=seq_length
- Decoder: n × 4(h×m + h²)
- Output: n × h×V where V=vocab_size
- Total: O(n[8hm + 8h² + hV])

Parameters:
- Embeddings: V × m
- Encoder LSTM: 4(h×m + h² + h)
- Decoder LSTM: 4(h×m + h² + h)  
- Output layer: h×V + V
- Total: V(m+1) + 8h(m+h+1) + hV

With default settings (m=128, h=256, V≈100):
- ~12M operations per word
- ~830K trainable parameters
"""