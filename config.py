# ===== Model architecture (must match training exactly) =====
MODEL_NAME = "miniGPT-TinyStories"
WEIGHTS_PATH = "models/tinystories_50m_model.weights.h5"

VOCAB_SIZE = 50257
D_MODEL = 768
NUM_HEADS = 12
HIDDEN_DIM = 3072
NUM_LAYERS = 5
CONTEXT_LENGTH = 256

TOKENIZER_NAME = "openai-community/gpt2"

# ===== Generation defaults =====
DEFAULT_MAX_NEW_TOKENS = 50
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

# ===== Server =====
HOST = "0.0.0.0"
PORT = 5000