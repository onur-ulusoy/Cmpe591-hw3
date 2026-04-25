# Hyperparameters for CNMP

# Model Architecture
HIDDEN_SIZE = 128
NUM_LAYERS = 4
CONTEXT_DIM = 6  # (t, ey, ez, oy, oz, h)
QUERY_DIM = 2    # (t, h)
TARGET_DIM = 4   # (ey, ez, oy, oz)
MIN_STD = 0.01

# Training
BATCH_SIZE = 16
NUM_ITERATIONS = 20000
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8

# Data Collection
NUM_TRAJECTORIES = 100
MAX_STEPS = 100

# File Paths
DATA_PATH = "trajectories.npy"
MODEL_PATH = "cnmp_model.pth"
LOG_PATH = "training.log"
