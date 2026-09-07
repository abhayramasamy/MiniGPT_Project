import time
import tensorflow as tf
from flask import Flask, request, jsonify, Response, stream_with_context

import config
from tokenizer import load_tokenizer
from load_weights import load_model
from inference import InferenceEngine
from stats import RuntimeStats

app = Flask(__name__)

# Model loaded ONCE at startup, not per-request.
print("Loading tokenizer...")
tokenizer = load_tokenizer()
print("Building model and loading weights...")
model = load_model()
print("Model ready.")

engine = InferenceEngine(model, tokenizer)
stats = RuntimeStats()
SERVER_START = time.time()


def _param_count():
    try:
        return int(sum(tf.size(w).numpy() for w in model.trainable_variables))
    except Exception:
        return None


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": config.MODEL_NAME})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", config.DEFAULT_MAX_NEW_TOKENS)
    temperature = data.get("temperature", config.DEFAULT_TEMPERATURE)
    top_p = data.get("top_p", config.DEFAULT_TOP_P)

    stats.start_request()

    def stream():
        metrics = None
        try:
            for event in engine.generate(prompt, max_new_tokens, temperature, top_p):
                if event["type"] == "token":
                    yield event["text"]
                else:
                    metrics = event["metrics"]
            stats.end_request(metrics=metrics)
        except Exception as e:
            stats.end_request(error=True)
            yield f"\n[error: {e}]"

    return Response(stream_with_context(stream()), mimetype="text/plain")


# ---------------- /dev namespace ----------------

@app.route("/dev/health", methods=["GET"])
def dev_health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model": config.MODEL_NAME,
        "parameter_count": _param_count(),
        "context_length": config.CONTEXT_LENGTH,
        "dtype": "float32",
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
        "uptime_seconds": round(time.time() - SERVER_START, 1),
    })


@app.route("/dev/stats", methods=["GET"])
def dev_stats():
    return jsonify(stats.snapshot())


@app.route("/dev/generate", methods=["POST"])
def dev_generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", config.DEFAULT_MAX_NEW_TOKENS)
    temperature = data.get("temperature", config.DEFAULT_TEMPERATURE)
    top_p = data.get("top_p", config.DEFAULT_TOP_P)

    stats.start_request()
    parts, metrics = [], None
    try:
        for event in engine.generate(prompt, max_new_tokens, temperature, top_p):
            if event["type"] == "token":
                parts.append(event["text"])
            else:
                metrics = event["metrics"]
    except Exception as e:
        stats.end_request(error=True)
        return jsonify({"error": str(e)}), 500

    stats.end_request(metrics=metrics)
    return jsonify({
        "text": "".join(parts),
        "metrics": metrics,
        "generation": {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens},
        "model": {"name": config.MODEL_NAME, "context_length": config.CONTEXT_LENGTH, "dtype": "float32"},
    })


@app.route("/dev/evaluate", methods=["POST"])
def dev_evaluate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    target = data.get("target", "")
    if not target:
        return jsonify({"error": "target is required"}), 400
    return jsonify(engine.evaluate(prompt, target))


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, threaded=True)