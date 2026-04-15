# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Please refer to README.md in the same folder for more information.

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from executorch.examples.models.llama.attention import (
    Attention,
    ATTENTION_REGISTRY,
    ForwardOptions,
)
from executorch.examples.models.llama.feed_forward import FeedForward
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.norm import RMSNorm
from executorch.examples.models.llama.rope import Rope
from torch import nn


class ConditionalFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        hidden_dim = args.hidden_dim
        if hidden_dim is None:
            # If hidden_dim is not explicitly set in the ModelArgs,
            # then calculate implicitly based on dim and also multiple of `args.multiple_of`
            multiple_of = args.multiple_of
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Parameter(torch.randn(args.num_experts, hidden_dim, self.dim))
        self.w2 = nn.Parameter(torch.randn(args.num_experts, hidden_dim, self.dim))
        self.w3 = nn.Parameter(torch.randn(args.num_experts, hidden_dim, self.dim))
        self.num_experts = args.num_experts

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        w1_weights = self.w1[expert_indices].transpose(-1, -2)  # [T, A, D, D]
        w3_weights = self.w3[expert_indices].transpose(-1, -2)  # [T, A, D, D]
        w2_weights = self.w2[expert_indices]  # [T, A, D, D]
        x1 = F.silu(torch.einsum("ti,taio -> tao", x, w1_weights))
        x3 = torch.einsum("ti, taio -> tao", x, w3_weights)
        expert_outs = torch.einsum("tao, taoi -> tai", (x1 * x3), w2_weights)
        return expert_outs


class MOEFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(config)
        self.dim = config.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.dim)
        # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
        # x: [T, D]
        scores = self.gate(x)  # [T, E]
        expert_weights, expert_indices = torch.topk(scores, 2, dim=-1)  # [T, A], [T, A]
        expert_weights = expert_weights.softmax(dim=-1)  # [T, A]
        expert_outs = self.cond_ffn(x, expert_indices)
        return torch.einsum("tai,ta -> ti", expert_outs, expert_weights)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, attention: Attention, layer_id: int = 0):
        """
        Transformer block with support for pre-norm and post-norm.
        Args:
            args (ModelArgs): model configuration parameters.
            attention (Attention): attention object to use in the transformer
                block. See `attention.py` for types of attention. Make sure
                the attention type is registered in the ATTENTION_REGISTRY.
            layer_id (int): layer index, used for per-layer dimension resolution.
        """
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.get_head_dim(layer_id)
        self.attention = attention

        hidden_dim = args.get_hidden_dim(layer_id)
        assert (
            hidden_dim is not None
        ), "`hidden_dim` must be set in ModelArgs to construct a TransformerBlock."
        if args.moe:
            self.block_sparse_moe = MOEFeedForward(args)
        else:
            self.feed_forward = FeedForward(
                dim=args.dim, hidden_dim=hidden_dim,
                act_fn=args.act_fn.get_function() if hasattr(args.act_fn, 'get_function') else None,
            )

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.post_attention_norm:
            self.post_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        if args.post_ffn_norm:
            self.post_ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # Per-layer embedding (Gemma-4): applied AFTER attention+FFN residuals
        self._has_per_layer_embed = False
        if hasattr(args, 'per_layer_embed_dim') and args.per_layer_embed_dim and args.per_layer_embed_dim > 0:
            pld = args.per_layer_embed_dim  # typically 256
            self.per_layer_input_gate = nn.Linear(args.dim, pld, bias=False)
            self.per_layer_projection = nn.Linear(pld, args.dim, bias=False)
            self.post_per_layer_input_norm = RMSNorm(args.dim, eps=args.norm_eps)
            # HF uses the model's activation function (gelu_approx), NOT sigmoid
            self._per_layer_act_fn = args.act_fn.get_function() if hasattr(args.act_fn, 'get_function') else F.silu
            self._has_per_layer_embed = True
            self._layer_id = layer_id

        # Layer scalar (Gemma-4): applied unconditionally at the end of each layer
        # Initialized to 1.0 (no-op for non-Gemma models); loaded from checkpoint
        self.layer_scalar = nn.Parameter(torch.ones(1))

    @classmethod
    def from_type(cls, layer_id, args, rope) -> "TransformerBlock":
        """
        Create a TransformerBlock with the legacy constructor.
        Args:
            layer_id (int): the index of the layer.
            args (ModelArgs): model configuration parameters.
            rope (Rope): the rope object to use for rotary embeddings.
        """
        if args.attention_type not in ATTENTION_REGISTRY:
            raise ValueError(
                f"Unknown attention type: {args.attention_type}. "
                f"Available: {list(ATTENTION_REGISTRY.keys())}"
            )
        cls = ATTENTION_REGISTRY[args.attention_type]
        attention = cls(args, layer_id, rope, **args.attention_kwargs)
        return TransformerBlock(args, attention)

    def forward(self, x, freqs_cos, freqs_sin, attn_options: ForwardOptions):  # x: 1xN
        h, attn_options_update = self.attention.forward(
            self.attention_norm(x), freqs_cos, freqs_sin, **attn_options
        )
        if hasattr(self, "post_attention_norm"):
            h = self.post_attention_norm(h)

        h = x + h
        if hasattr(self, "block_sparse_moe"):
            ffn_out = self.block_sparse_moe(self.ffn_norm(h))
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
        if hasattr(self, "post_ffn_norm"):
            ffn_out = self.post_ffn_norm(ffn_out)
        out = h + ffn_out

        # Per-layer embedding injection (Gemma-4) — AFTER attention+FFN residuals
        # HF flow: gate → act_fn → multiply with per_layer_input → project → norm → residual
        if self._has_per_layer_embed:
            per_layer_emb = attn_options.get("_per_layer_embs")
            if per_layer_emb is not None:
                residual = out
                gated = self._per_layer_act_fn(self.per_layer_input_gate(out))
                projected = self.per_layer_projection(gated * per_layer_emb)
                out = residual + self.post_per_layer_input_norm(projected)

        # Layer scalar (Gemma-4): unconditional scaling at end of layer
        out = out * self.layer_scalar

        return out, attn_options_update


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, layers: nn.ModuleList, rope: Rope):
        """
        Transformer model.
        Args:
            params (ModelArgs): model configuration parameters.
            layers (nn.ModuleList): list of transformer blocks - see the
                `TransformerBlock` type above.
            rope (Rope): the rope object to use for rotary embeddings.
        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.apply_embedding = params.apply_embedding
        self.apply_output = params.apply_output

        self.tok_embeddings = (
            nn.Embedding(params.vocab_size, params.dim)
            if self.apply_embedding
            else None
        )
        self.layers = layers
        self.rope = rope
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = (
            nn.Linear(params.dim, params.vocab_size, bias=False)
            if self.apply_output
            else None
        )
        self.use_kv_cache = params.use_kv_cache
        self.generate_full_logits = params.generate_full_logits
        self.max_seq_len = params.max_seq_len
        self.max_context_len = params.max_context_len
        self.input_prune_map = params.input_prune_map
        self.output_prune_map = params.output_prune_map

        # Per-layer embedding table (Gemma-4): [vocab, n_layers * per_layer_embed_dim]
        self._per_layer_embed_dim = getattr(params, 'per_layer_embed_dim', 0) or 0
        self._kv_donor_map = params.kv_donor_map or {}
        self._final_logit_softcapping = getattr(params, 'final_logit_softcapping', 0.0)
        if self._per_layer_embed_dim > 0:
            pld = self._per_layer_embed_dim
            self.embed_tokens_per_layer = nn.Embedding(
                params.vocab_size, params.n_layers * pld
            )
            # HF Gemma4: per_layer_model_projection projects main embed to per-layer space
            self.per_layer_model_projection = nn.Linear(
                params.dim, params.n_layers * pld, bias=False
            )
            self._per_layer_model_projection_scale = params.dim ** -0.5
            self.per_layer_projection_norm = RMSNorm(pld, eps=params.norm_eps)
            self._per_layer_input_scale = 2.0 ** -0.5
            self._per_layer_embed_scale = pld ** 0.5  # ScaledWordEmbedding scale

    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,  # tokens
        attn_options: Optional[ForwardOptions] = None,
        h: Optional[torch.FloatTensor] = None,  # embeddings
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Any]]]:
        if (tokens is None) ^ (h is not None):
            raise ValueError(
                "You cannot specify both tokens and h at the same time, and must specify either one"
            )
        if self.apply_embedding and tokens is not None and h is None:
            h = self.tok_embeddings(tokens)
            if self.params.embedding_scale_factor != 1.0:
                h = h * self.params.embedding_scale_factor

        if attn_options is None:
            attn_options = {}
        seqlen = h.shape[1]
        freqs_cos, freqs_sin = self.rope.get_freqs(
            attn_options.get("input_pos"), seqlen
        )

        # Make a shallow copy so the updates don't get captured by export
        attn_options_ = attn_options.copy() if attn_options is not None else {}
        attn_options_update = None

        # Per-layer embedding (Gemma-4): combine token-level + projection-level embeddings
        per_layer_embs_all = None
        if self._per_layer_embed_dim > 0 and hasattr(self, 'embed_tokens_per_layer') and tokens is not None:
            pld = self._per_layer_embed_dim
            n_layers = self.n_layers
            bsz_s = tokens.shape

            # Step 1: per-layer token embeddings (scaled)
            per_layer_tok = self.embed_tokens_per_layer(tokens)  # [B, S, n_layers * pld]
            per_layer_tok = per_layer_tok * self._per_layer_embed_scale
            per_layer_tok = per_layer_tok.view(bsz_s[0], bsz_s[1], n_layers, pld)

            # Step 2: project main embeddings to per-layer space
            per_layer_proj = self.per_layer_model_projection(h) * self._per_layer_model_projection_scale
            per_layer_proj = per_layer_proj.view(bsz_s[0], bsz_s[1], n_layers, pld)
            per_layer_proj = self.per_layer_projection_norm(per_layer_proj)

            # Step 3: combine
            per_layer_embs_all = (per_layer_proj + per_layer_tok) * self._per_layer_input_scale

        # Collect donor KV caches for shared-KV layers
        donor_caches = {}

        for layer_idx, layer in enumerate(self.layers):
            # Inject per-layer embedding for this layer
            if per_layer_embs_all is not None:
                attn_options_["_per_layer_embs"] = per_layer_embs_all[:, :, layer_idx, :]
            else:
                attn_options_.pop("_per_layer_embs", None)

            # Pass donor caches for shared KV layers
            attn_options_["_donor_caches"] = donor_caches

            h, attn_options_update = layer(h, freqs_cos, freqs_sin, attn_options_)
            if attn_options_update is not None:
                attn_options_.update(**attn_options_update)

            # After forward, store this layer's K/V for potential donor access
            attn = layer.attention if hasattr(layer, 'attention') else None
            if attn is not None and attn._kv_donor_id is None:
                if self.use_kv_cache and hasattr(attn, 'kv_cache') and attn.kv_cache is not None:
                    donor_caches[layer_idx] = (attn.kv_cache.k_cache, attn.kv_cache.v_cache)
                elif hasattr(attn, '_last_k'):
                    donor_caches[layer_idx] = (attn._last_k, attn._last_v)

        if not self.generate_full_logits:
            # Only the last logit is used for the new generated token
            pos = attn_options.get("last_valid_token_pos", -1)
            h = h[:, pos, :]

        h = self.norm(h)

        if self.apply_output:
            logits = self.output(h)

            # Gemma-4: final logit softcapping → tanh(logits/cap) * cap
            if self._final_logit_softcapping > 0:
                cap = self._final_logit_softcapping
                logits = torch.tanh(logits / cap) * cap

            if self.output_prune_map is not None:
                # expand to original size so that downstream applications can use the logits as-is.
                if self.generate_full_logits:
                    # (1, seq_len, pruned_size) -> (1, seq_len, original_size)
                    expanded_logits = torch.full(
                        [logits.shape[0], logits.shape[1], self.vocab_size],
                        float("-inf"),
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                    expanded_logits[:, :, list(self.output_prune_map.values())] = logits
                else:
                    # (1, pruned_size) -> (1, original_size)
                    expanded_logits = torch.full(
                        [logits.shape[0], self.vocab_size],
                        float("-inf"),
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                    expanded_logits[:, list(self.output_prune_map.values())] = logits
                logits = expanded_logits
        else:
            logits = h

        if attn_options_update is not None:
            return logits, attn_options_update

        return logits


def construct_transformer(model_args: ModelArgs) -> Transformer:
    """
    Construct a Transformer model from the given model arguments.
    """
    rope = Rope(model_args)
    if model_args.attention_type not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention type: {model_args.attention_type}. "
            f"Available: {list(ATTENTION_REGISTRY.keys())}"
        )
    layers = torch.nn.ModuleList()
    cls = ATTENTION_REGISTRY[model_args.attention_type]
    for layer_id in range(model_args.n_layers):
        # hybrid models define layer_types
        if model_args.layer_types and model_args.layer_types[layer_id] == "conv":
            from executorch.examples.models.lfm2.short_conv import ShortConvBlock

            assert (
                model_args.hidden_dim is not None
            ), "`hidden_dim` must be set in ModelArgs to construct a TransformerBlock."
            layers.append(
                ShortConvBlock(
                    dim=model_args.dim,
                    hidden_dim=model_args.hidden_dim,
                    norm_eps=model_args.norm_eps,
                )
            )
        else:
            attention = cls(
                model_args, layer_id, rope, **model_args.attention_kwargs
            )  # pyre-ignore[45]
            transformer_block = TransformerBlock(model_args, attention, layer_id)
            layers.append(transformer_block)

    # Wire donor KVCache references for shared-KV layers (export-safe: avoids dict passing)
    if model_args.kv_donor_map and model_args.use_kv_cache:
        for layer_id in range(model_args.n_layers):
            layer = layers[layer_id]
            attn = getattr(layer, 'attention', None)
            if attn is not None and attn._kv_donor_id is not None:
                donor_attn = layers[attn._kv_donor_id].attention
                # Plain Python attr (not nn.Module child) so donor cache isn't re-registered
                object.__setattr__(attn, '_donor_kv_cache_ref', donor_attn.kv_cache)

    return Transformer(model_args, layers, rope)
