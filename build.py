from tensorflow import keras
import config
from model import RMSNorm, Decoder_block, LMHead


def build_model():
    # Embedding is built with a throwaway initializer — the trained values live
    # in the checkpoint and get restored by load_weights() in load_weights.py.
    # This is why no PyTorch/GPT-2 download is needed at inference time.
    embedding = keras.layers.Embedding(
        input_dim=config.VOCAB_SIZE,
        output_dim=config.D_MODEL,
        trainable=False,
        name="embedding",
    )

    decoder_blocks = [
        Decoder_block(d_model=config.D_MODEL, num_heads=config.NUM_HEADS, hidden_dim=config.HIDDEN_DIM)
        for _ in range(config.NUM_LAYERS)
    ]

    model = keras.models.Sequential([
        embedding,
        *decoder_blocks,
        RMSNorm(config.D_MODEL),
        LMHead(embedding_layer=embedding, temperature=8, return_last_token=False),
    ])

    model.build(input_shape=(None, config.CONTEXT_LENGTH))
    return model