import time
import numpy as np
import tensorflow as tf
import config


def _nucleus_filter(logits, top_p):
    probs = tf.nn.softmax(logits).numpy()
    sorted_idx = np.argsort(-probs)
    cumulative = np.cumsum(probs[sorted_idx])
    cutoff = np.searchsorted(cumulative, top_p) + 1
    nucleus_idx = sorted_idx[:cutoff]
    nucleus_probs = probs[nucleus_idx]
    return nucleus_idx, nucleus_probs / nucleus_probs.sum()


class InferenceEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _logits_for(self, ids):
        x = tf.constant([ids], dtype=tf.int32)
        return self.model(x, training=False).numpy()[0]  # (seq_len, vocab)

    def generate(self, prompt, max_new_tokens=None, temperature=None, top_p=None):
        """Generator: yields {'type':'token','text':...} pieces, then one {'type':'done','metrics':...}."""
        max_new_tokens = max_new_tokens or config.DEFAULT_MAX_NEW_TOKENS
        temperature = max(temperature or config.DEFAULT_TEMPERATURE, 1e-6)
        top_p = top_p or config.DEFAULT_TOP_P

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_token_count = len(input_ids)

        t_start = time.perf_counter()
        first_token_time = None
        output_count = 0

        for _ in range(max_new_tokens):
            context = input_ids[-config.CONTEXT_LENGTH:]
            logits = self._logits_for(context)[-1] / temperature

            idx, probs = _nucleus_filter(logits, top_p)
            next_id = int(np.random.choice(idx, p=probs))

            if first_token_time is None:
                first_token_time = time.perf_counter()

            input_ids.append(next_id)
            output_count += 1

            yield {"type": "token", "text": self.tokenizer.decode([next_id], skip_special_tokens=True)}

        t_end = time.perf_counter()
        gen_ms = (t_end - t_start) * 1000
        ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else None

        yield {
            "type": "done",
            "metrics": {
                "input_tokens": input_token_count,
                "output_tokens": output_count,
                "total_tokens": input_token_count + output_count,
                "time_to_first_token_ms": round(ttft_ms, 2) if ttft_ms else None,
                "generation_time_ms": round(gen_ms, 2),
                "tokens_per_second": round(output_count / (gen_ms / 1000), 2) if gen_ms > 0 else 0.0,
            },
        }

    def evaluate(self, prompt, target):
        """Teacher-forced perplexity of `target` continuing `prompt` — NOT the same as
        perplexity of a sampled generation; this scores the model against a fixed continuation."""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)

        sequence = (prompt_ids + target_ids)[-(config.CONTEXT_LENGTH + 1):]
        logits = self._logits_for(sequence[:-1])  # (seq_len-1, vocab)

        offset = len(sequence) - 1 - len(target_ids)
        token_logprobs = []
        total_loss = 0.0
        for i, tgt_id in enumerate(target_ids):
            row = logits[offset + i]
            logprobs = row - tf.reduce_logsumexp(row).numpy()
            lp = float(logprobs[tgt_id])
            token_logprobs.append(round(lp, 4))
            total_loss += -lp

        avg_loss = total_loss / max(len(target_ids), 1)
        return {
            "input_tokens": len(prompt_ids),
            "target_tokens": len(target_ids),
            "loss": round(avg_loss, 4),
            "perplexity": round(float(np.exp(avg_loss)), 4),
            "token_logprobs": token_logprobs,
        }