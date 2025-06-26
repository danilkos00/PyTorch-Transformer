import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float, Int
from collections import OrderedDict
from einops import einsum, rearrange
from .tokenizer import Tokenizer
from .nn_tools import softmax
from .flash_attention import PytorchFlashAttention2Func


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """A linear layer initialized with truncated normal fan-in fan-out.

        Args:
            in_features: int
                The number of input features.
            out_features: int
                The number of output features.
        """
        super().__init__()

        std = (2 / (in_features + out_features)) ** 0.5

        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(out_features, in_features), std=std, a=-3*std, b=3*std)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()

        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim), a=-3, b=3)
        )


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    """
    This module implements root mean square layer normalization

    Args:
        d_model: int
            Dimensionality of the input to normalize.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.

    Returns:
        FloatTensor of same shape as input.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return (x.mul(rms).mul(self.weight)).to(in_dtype)


class PositionWiseFFNet(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()

        if d_ff is None:
            d_ff_raw = 8 / 3 * d_model
            d_ff = int(-1 * d_ff_raw // 64 * -1 * 64)

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)


    def _silu(self, x):
        return x.mul(torch.sigmoid(x))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj1 = self.w1(x)
        proj3 = self.w3(x)

        return self.w2(self._silu(proj1).mul(proj3))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()

        freqs = 1.0 / theta ** (2 * torch.arange(0, d_k // 2) / d_k)

        positions = torch.arange(max_seq_len)

        angles = einsum(positions, freqs, 'max_seq_len, d_k -> max_seq_len d_k')

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_vals = self.cos[token_positions] # (..., seq_len, d_k//2)
        sin_vals = self.sin[token_positions] # (..., seq_len, d_k//2)

        x_first, x_second = rearrange(x, '... (half_d_k xy) -> xy ... half_d_k', xy=2)

        x_rotated = torch.empty_like(x, dtype=x.dtype)

        x_rotated[..., ::2] = x_first.mul(cos_vals) - x_second.mul(sin_vals)
        x_rotated[..., 1::2] = x_first.mul(sin_vals) + x_second.mul(cos_vals)

        return x_rotated.contiguous()


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    is_causal: bool = True
) -> Float[Tensor, " ... queries d_v"]:
    scale = 1 / Q.size(-1)**0.5

    attention_scores = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys') * scale

    if is_causal:
        mask = torch.ones_like(attention_scores, dtype=torch.bool).tril()
        attention_scores = torch.where(mask, attention_scores, float('-inf'))

    attention_scores = einsum(softmax(attention_scores, dim=-1), V, '... queries seq_len, ... seq_len d_v -> ... queries d_v')

    return attention_scores


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_flash_attention: bool, rope_class: RotaryPositionalEmbedding):
        super().__init__()

        self.num_heads = num_heads

        self.rope = rope_class

        if use_flash_attention:
            self.attention_function = PytorchFlashAttention2Func.apply
        else:
            self.attention_function = scaled_dot_product_attention

        self.qkv_layer = Linear(d_model, 3 * d_model)
        self.output_layer = Linear(d_model, d_model)


    def forward(self,
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"]
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        Q, K, V = self.qkv_layer(in_features).chunk(3, dim=-1)

        Q = rearrange(Q, '... seq_len (h d_k) -> ... h seq_len d_k', h=self.num_heads)
        K = rearrange(K, '... seq_len (h d_k) -> ... h seq_len d_k', h=self.num_heads)
        V = rearrange(V, '... seq_len (h d_k) -> ... h seq_len d_k', h=self.num_heads)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        attention_output = self.attention_function(Q, K, V, True)

        attention_output = rearrange(attention_output, '... h seq_len d_v -> ... seq_len (h d_v)')

        return self.output_layer(attention_output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int | None, use_flash_attention, rope: RotaryPositionalEmbedding):
        super().__init__()

        self.multi_head_self_attention = MultiHeadSelfAttention(d_model, num_heads, use_flash_attention, rope)

        self.rms_norm1 = RMSNorm(d_model)
        self.rms_norm2 = RMSNorm(d_model)

        self.ffn = PositionWiseFFNet(d_model, d_ff)


    def forward(self, in_features: Float[Tensor, " batch sequence_length d_model"]):
        token_positions = torch.arange(in_features.size(-2), device=in_features.device).unsqueeze(0)

        sublayer_output = in_features + self.multi_head_self_attention(self.rms_norm1(in_features), token_positions)

        return sublayer_output + self.ffn(self.rms_norm2(sublayer_output))


class TransformerLM(nn.Module):
    """A Transformer language model.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int | None, default is None
            Dimensionality of the feed-forward inner layer. If value is None
            d_ff = 8/3 * d_model
        rope_theta: float, default is 10000.0
            The theta value for the RoPE positional encoding.

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the
        predicted unnormalized next-word distribution for each token.
    """
    def __init__(self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int| None = None,
        rope_theta: float = 10000.0,
        use_flash_attention: bool = False
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model)

        rope_class = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)

        self.transformer_blocks = nn.Sequential(
            OrderedDict([
                (f'tf_block_{i + 1}', TransformerBlock(d_model, num_heads, d_ff, use_flash_attention, rope_class))
                for i in range(num_layers)
            ])
        )

        self.output_norm = RMSNorm(d_model)

        self.lm_head = Linear(d_model, vocab_size)


    def forward(self,
        in_indices: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        input_embeddings = self.token_embeddings(in_indices)

        blocks_output = self.transformer_blocks(input_embeddings)

        blocks_output = self.output_norm(blocks_output)

        output_embeddings = self.lm_head(blocks_output)

        return output_embeddings


    def generate(self, prompt: str, tokenizer : Tokenizer, max_tokens: int, p: float = 0.9, temperature: float = 1.0):
        previous_state = self.training

        self.eval()

        device = next(self.parameters()).device

        print(prompt, end='')

        tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)

        for _ in range(max_tokens):
            with torch.no_grad():
                logits = self.forward(tokens.unsqueeze(0))

            next_token_probs = softmax(logits[0, -1, :], tau=temperature)

            probs, ids = next_token_probs.sort(descending=True)

            prob_sum = 0.0
            top_p_probs = torch.zeros_like(next_token_probs)
            for i in range(len(probs)):
                if prob_sum >= p:
                    break
                prob_sum += probs[i].item()
                top_p_probs[ids[i]] = probs[i].item()

            top_p_probs /= prob_sum

            next_token_idx = torch.multinomial(top_p_probs, 1).item()

            next_token = tokenizer.vocab[next_token_idx].decode('utf-8', errors='replace')

            if next_token == '<|endoftext|>':
                break

            print(next_token, end='')

            if len(tokens) < self.context_length:
                tokens = torch.cat([tokens, torch.tensor([next_token_idx], device=device)])
            else:
                tokens = torch.cat([tokens[1:], torch.tensor([next_token_idx], device=device)])

        self.train(previous_state)
