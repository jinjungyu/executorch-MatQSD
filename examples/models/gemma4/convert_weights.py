"""Convert Gemma-4 E2B HuggingFace weights to ExecuTorch format.

Handles Gemma-4's non-uniformities:
  - Double-wide MLP (layers >= 15)
  - Hybrid attention (sliding head_dim=256, full head_dim=512)
  - Shared KV layers (layers >= 15): duplicates donor KV weights
  - Per-layer embedding: SKIPPED in v1
  - tie_word_embeddings: output.weight = tok_embeddings.weight

Usage:
    python convert_weights.py <hf_model_dir> <output.pth>
"""

import argparse
import json
import os
from typing import Dict

import torch


# Gemma-4 layer types: [sliding x4, full] x 7 = 35 layers
_LAYER_TYPES = (["sliding_attention"] * 4 + ["full_attention"]) * 7
_N_LAYERS = 35
_NUM_KV_SHARED_LAYERS = 20  # layers 15-34 share KV from layers 0-14


# HF key -> ET key mapping
# Gemma-4 uses "model.language_model.layers" prefix (multimodal wrapper)
_HF_TO_ET = {
    "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
    "model.language_model.norm.weight": "norm.weight",
    "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.language_model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm_fn.weight",
    "model.language_model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm_fn.weight",
    "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_norm.weight",
    "model.language_model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.language_model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.post_ffn_norm.weight",
    "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    # Per-layer embedding
    "model.language_model.layers.{}.per_layer_input_gate.weight": "layers.{}.per_layer_input_gate.weight",
    "model.language_model.layers.{}.per_layer_projection.weight": "layers.{}.per_layer_projection.weight",
    "model.language_model.layers.{}.post_per_layer_input_norm.weight": "layers.{}.post_per_layer_input_norm.weight",
    "model.language_model.layers.{}.layer_scalar": "layers.{}.layer_scalar",
}

# Special keys (not per-layer indexed)
_HF_TO_ET_SPECIAL = {
    "model.language_model.embed_tokens_per_layer.weight": "embed_tokens_per_layer.weight",
    "model.language_model.per_layer_model_projection.weight": "per_layer_model_projection.weight",
    "model.language_model.per_layer_projection_norm.weight": "per_layer_projection_norm.weight",
}


def _get_mapped_key(key: str) -> str:
    """Map HF key to ET key, handling layer indices."""
    # Check special (non-layer-indexed) keys first
    if key in _HF_TO_ET_SPECIAL:
        return _HF_TO_ET_SPECIAL[key]
    for hf_pat, et_pat in _HF_TO_ET.items():
        if "{}" in hf_pat:
            prefix = hf_pat.split("{}")[0]
            suffix = hf_pat.split("{}")[1]
            if key.startswith(prefix) and key.endswith(suffix):
                rest = key[len(prefix):]
                layer_id = rest[:rest.index(".") if "." in rest else len(rest)]
                return et_pat.replace("{}", layer_id)
        elif key == hf_pat:
            return et_pat
    return None


def _kv_donor_layer(layer_id: int) -> int:
    """For shared-KV layers (>= 15), find the donor layer.
    HF logic: each shared layer uses the LAST non-shared layer of the same type.
    For Gemma-4 [s,s,s,s,f]×7: all shared sliding → layer 13, all shared full → layer 14.
    """
    if layer_id < _N_LAYERS - _NUM_KV_SHARED_LAYERS:
        return layer_id  # Not shared
    # Determine layer type
    layer_type = _LAYER_TYPES[layer_id]
    if layer_type == "full_attention":
        return 14  # Last non-shared full attention layer
    else:
        return 13  # Last non-shared sliding attention layer


def gemma4_to_executorch(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert HF Gemma-4 state dict to ExecuTorch format."""
    converted = {}
    skipped = []

    for key, value in state_dict.items():
        # Skip multimodal-only projection (embed_vision, embed_audio)
        if key.startswith("model.embed_vision") or key.startswith("model.embed_audio"):
            skipped.append(key)
            continue
        # Skip audio/vision components
        if key.startswith("audio_") or key.startswith("vision_") or key.startswith("multi_modal"):
            skipped.append(key)
            continue

        et_key = _get_mapped_key(key)
        if et_key is not None:
            converted[et_key] = value
        else:
            skipped.append(key)

    # Shared KV layers (>= 15): do NOT duplicate KV weights.
    # These layers have no wk/wv modules — they use donor K/V at runtime.

    # tie_word_embeddings: output.weight = tok_embeddings.weight
    if "tok_embeddings.weight" in converted and "output.weight" not in converted:
        converted["output.weight"] = converted["tok_embeddings.weight"]

    print(f"[Gemma-4] Converted {len(converted)} keys, skipped {len(skipped)} keys")
    if skipped:
        print(f"  Skipped examples: {skipped[:5]}")

    return converted


def load_checkpoint(input_dir: str) -> Dict:
    """Load HF checkpoint (safetensors or pytorch_model.bin)."""
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        from safetensors.torch import load_file
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shards = sorted(set(weight_map.values()))
        merged = {}
        for shard in shards:
            merged.update(load_file(os.path.join(input_dir, shard)))
        return merged

    st_path = os.path.join(input_dir, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        return load_file(st_path)

    pt_path = os.path.join(input_dir, "pytorch_model.bin")
    if os.path.exists(pt_path):
        return torch.load(pt_path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(f"No checkpoint found in {input_dir}")


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading Gemma-4 checkpoint...")
    sd = load_checkpoint(input_dir)
    print("Converting to ExecuTorch format...")
    sd = gemma4_to_executorch(sd)
    print(f"Saving to {output_file}...")
    torch.save(sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemma-4 E2B weights to ExecuTorch format."
    )
    parser.add_argument("input_dir", type=str, help="HF model directory")
    parser.add_argument("output", type=str, help="Output .pth file")
    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
