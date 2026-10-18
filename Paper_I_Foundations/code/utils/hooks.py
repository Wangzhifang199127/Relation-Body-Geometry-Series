# code/utils/hooks.py
"""
QKV extraction hook utilities.
Provides a unified interface for extracting Q/K/V from the attention layers of GPT-2 (and compatible models).
"""

import torch
import numpy as np


class QKVHook:
    """
    Hook for extracting the output of c_attn.
    In GPT-2, attn.c_attn returns a tensor of shape [batch, seq, 3*n_embd],
    where the first n_embd is Q, the middle is K, and the last is V.
    """

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.output = None

    def __call__(self, module, inp, out):
        self.output = out.detach()

    def get_qkv(self, n_embd, n_head, head_dim, batch_idx=0):
        """
        Return Q, K, V with shape [n_head, seq, head_dim].
        """
        if self.output is None:
            raise RuntimeError("Hook has not captured output; run a forward pass first.")

        qkv = self.output[batch_idx]  # [seq, 3*n_embd]
        seq_len = qkv.shape[0]

        q, k, v = qkv.split(n_embd, dim=-1)
        q = q.view(seq_len, n_head, head_dim).transpose(0, 1)
        k = k.view(seq_len, n_head, head_dim).transpose(0, 1)
        v = v.view(seq_len, n_head, head_dim).transpose(0, 1)

        return q, k, v


class AttentionOutputHook:
    """
    Hook for extracting the attention output (the weighted result).
    Used for ablation experiments.
    """

    def __init__(self, layer_idx, head_indices=None):
        self.layer_idx = layer_idx
        self.head_indices = head_indices  # None means the whole layer
        self.enabled = False

    def __call__(self, module, inp, out):
        if not self.enabled:
            return out

        if isinstance(out, tuple):
            attn_output = out[0].clone()
            rest = out[1:]
        else:
            attn_output = out.clone()
            rest = None

        # Handle ablation logic
        if self.head_indices is None:
            attn_output[:, :, :] = 0.0
        else:
            for h in self.head_indices:
                start = h * (attn_output.shape[-1] // max(1, self._n_head))
                end = start + (attn_output.shape[-1] // max(1, self._n_head))
                attn_output[:, :, start:end] = 0.0

        if rest is not None:
            return (attn_output,) + rest
        return attn_output


class PatchHook:
    """
    Activation patching hook: replace the output of a specified head with its mean across all tokens.
    """

    def __init__(self, layer_idx, head_idx, head_dim):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.head_dim = head_dim
        self.enabled = False

    def __call__(self, module, inp, out):
        if not self.enabled:
            return out

        if isinstance(out, tuple):
            attn_output = out[0].clone()
            rest = out[1:]
        else:
            attn_output = out.clone()
            rest = None

        start = self.head_idx * self.head_dim
        end = start + self.head_dim
        mean_val = attn_output[:, :, start:end].mean(dim=1, keepdim=True)
        attn_output[:, :, start:end] = mean_val

        if rest is not None:
            return (attn_output,) + rest
        return attn_output


def register_qkv_hooks(model, n_layer):
    """
    Register QKV hooks for all layers. Returns a list of hooks and a list of handles.
    """
    hooks = []
    handles = []
    for l in range(n_layer):
        h = QKVHook(l)
        handle = model.transformer.h[l].attn.c_attn.register_forward_hook(h)
        hooks.append(h)
        handles.append(handle)
    return hooks, handles


def remove_hooks(handles):
    """Remove all hook handles."""
    for handle in handles:
        handle.remove()