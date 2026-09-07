import time


class RuntimeStats:
    def __init__(self):
        self.start_time = time.time()
        self.requests_total = 0
        self.errors_total = 0
        self.active_requests = 0
        self.total_tokens_generated = 0
        self._completed = 0
        self._in_sum = 0
        self._out_sum = 0
        self._ttft_sum = 0.0
        self._ttft_n = 0
        self._latency_sum = 0.0
        self._tps_sum = 0.0

    def start_request(self):
        self.requests_total += 1
        self.active_requests += 1

    def end_request(self, metrics=None, error=False):
        self.active_requests = max(0, self.active_requests - 1)
        if error:
            self.errors_total += 1
            return
        if not metrics:
            return
        self._completed += 1
        self.total_tokens_generated += metrics.get("output_tokens", 0)
        self._in_sum += metrics.get("input_tokens", 0)
        self._out_sum += metrics.get("output_tokens", 0)
        self._latency_sum += metrics.get("generation_time_ms", 0)
        self._tps_sum += metrics.get("tokens_per_second", 0)
        if metrics.get("time_to_first_token_ms") is not None:
            self._ttft_sum += metrics["time_to_first_token_ms"]
            self._ttft_n += 1

    def snapshot(self):
        n = max(self._completed, 1)
        return {
            "requests_total": self.requests_total,
            "errors_total": self.errors_total,
            "active_requests": self.active_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "average_input_tokens": round(self._in_sum / n, 2),
            "average_output_tokens": round(self._out_sum / n, 2),
            "average_tokens_per_second": round(self._tps_sum / n, 2),
            "average_time_to_first_token_ms": round(self._ttft_sum / max(self._ttft_n, 1), 2),
            "average_generation_latency_ms": round(self._latency_sum / n, 2),
            "uptime_seconds": round(time.time() - self.start_time, 1),
        }