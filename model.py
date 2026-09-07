import tensorflow as tf
from tensorflow import keras


class RMSNorm(keras.layers.Layer):
    def __init__(self, dem, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.gamma = self.add_weight(
            name="gamma", shape=(dem,), initializer="ones",
            trainable=True, dtype="float32",
        )

    def call(self, x):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        variance = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        x = x * tf.math.rsqrt(variance + self.eps)
        out = x * self.gamma
        return tf.cast(out, orig_dtype)


class RoPEencode(keras.layers.Layer):
    def __init__(self, max_seq_len, head_dim, base=10000, how="half_split", **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.base = base
        self.how = how

        i = tf.range(head_dim // 2, dtype=tf.float32)
        self.freq = tf.pow(tf.cast(base, tf.float32), -2 * i / head_dim)
        self.freq = tf.expand_dims(self.freq, axis=0)

        self.pos = tf.range(max_seq_len, dtype=tf.float32)
        self.pos = tf.expand_dims(self.pos, axis=1)
        self.theta = tf.matmul(self.pos, self.freq)

        if how == "half_split":
            self.theta2 = tf.concat([self.theta, self.theta], axis=-1)
        elif how == "interleaf":
            self.theta2 = tf.repeat(self.theta, repeats=2, axis=1)
        else:
            raise ValueError("unknown rope method")

        self.sine = tf.sin(self.theta2)
        self.cosine = tf.cos(self.theta2)
        self.sine = tf.expand_dims(tf.expand_dims(self.sine, axis=0), axis=0)
        self.cosine = tf.expand_dims(tf.expand_dims(self.cosine, axis=0), axis=0)

    def call(self, x):
        s = tf.shape(x)[2]
        d = tf.shape(x)[-1]

        if self.how == "half_split":
            rot_x = tf.concat([-x[:, :, :, d // 2:], x[:, :, :, :d // 2]], axis=-1)
        elif self.how == "interleaf":
            even = x[..., ::2]
            odd = x[..., 1::2]
            rot_x = tf.reshape(tf.stack([-odd, even], axis=-1), tf.shape(x))

        cos = tf.cast(self.cosine[:, :, :s, :], x.dtype)
        sin = tf.cast(self.sine[:, :, :s, :], x.dtype)
        return x * cos + rot_x * sin


class QKVProjection(keras.layers.Layer):
    def __init__(self, d_model=768, **kwargs):
        super().__init__(**kwargs)
        self.q_proj = keras.layers.Dense(d_model, use_bias=False)
        self.k_proj = keras.layers.Dense(d_model, use_bias=False)
        self.v_proj = keras.layers.Dense(d_model, use_bias=False)

    def call(self, x):
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)


def split_heads(x, num_heads=12):
    B = tf.shape(x)[0]
    t = tf.shape(x)[1]
    d = tf.shape(x)[2]
    head_dim = d // num_heads
    x = tf.reshape(x, [B, t, num_heads, head_dim])
    return tf.transpose(x, [0, 2, 1, 3])


class MHCrossAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads=12, max_seq_len=2048, rope_how="half_split", **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = QKVProjection(self.d_model)
        self.rope_q = RoPEencode(max_seq_len=max_seq_len, head_dim=self.head_dim, how=rope_how)
        self.rope_k = RoPEencode(max_seq_len=max_seq_len, head_dim=self.head_dim, how=rope_how)
        self.output_projection = keras.layers.Dense(self.d_model, use_bias=False)

    def call(self, x):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        q, k, v = self.qkv(x)

        q = split_heads(q, num_heads=self.num_heads)
        k = split_heads(k, num_heads=self.num_heads)
        v = split_heads(v, num_heads=self.num_heads)

        q = self.rope_q(q)
        k = self.rope_k(k)

        q_f32 = tf.cast(q, tf.float32)
        k_f32 = tf.cast(k, tf.float32)
        v_f32 = tf.cast(v, tf.float32)

        scores = tf.matmul(q_f32, k_f32, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        causal_mask = tf.linalg.band_part(tf.ones((T, T), dtype=tf.float32), -1, 0)
        causal_mask = tf.where(causal_mask == 0, tf.constant(-1e9, dtype=tf.float32), 0.0)
        scores = scores + causal_mask[tf.newaxis, tf.newaxis, :, :]
        scores = tf.nn.softmax(scores, axis=-1)

        out = tf.matmul(scores, v_f32)
        out = tf.transpose(out, [0, 2, 1, 3])
        tq = tf.shape(out)[1]
        out = tf.reshape(out, [B, tq, self.d_model])
        out = self.output_projection(out)
        return tf.cast(out, x.dtype)


class SwiGLU(keras.layers.Layer):
    def __init__(self, d_model, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.gate_proj = keras.layers.Dense(hidden_dim, use_bias=False, name="gate_proj")
        self.up_proj = keras.layers.Dense(hidden_dim, use_bias=False, name="up_proj")
        self.down_proj = keras.layers.Dense(d_model, use_bias=False, name="down_proj")

    def call(self, inputs):
        gate = tf.nn.silu(self.gate_proj(inputs))
        up = self.up_proj(inputs)
        return self.down_proj(gate * up)


class Decoder_block(keras.layers.Layer):
    def __init__(self, d_model, num_heads, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.norm1 = RMSNorm(self.d_model)
        self.attn = MHCrossAttention(d_model=self.d_model, num_heads=self.num_heads)
        self.norm2 = RMSNorm(self.d_model)
        self.ffn = SwiGLU(d_model=self.d_model, hidden_dim=self.hidden_dim)

    def call(self, x):
        x_1 = self.norm1(x)
        x = x + self.attn(x_1)
        x_2 = self.norm2(x)
        x = x + self.ffn(x_2)
        return x


class LMHead(keras.layers.Layer):
    def __init__(self, embedding_layer, temperature=8, return_last_token=False, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.return_last_token = return_last_token

    def call(self, inputs):
        if self.return_last_token:
            inputs = inputs[:, -1, :]
        inputs = tf.cast(inputs, tf.float32)
        weights = tf.cast(self.embedding_layer.embeddings, tf.float32)
        return tf.matmul(inputs, weights, transpose_b=True)  # raw logits