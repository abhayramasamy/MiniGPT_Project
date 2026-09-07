import os
import config
from build import build_model


def load_model():
    model = build_model()
    if not os.path.exists(config.WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Weights not found at {config.WEIGHTS_PATH}. "
            f"Place tinystories_50m_model.weights.h5 inside the models/ directory."
        )
    model.load_weights(config.WEIGHTS_PATH)
    return model