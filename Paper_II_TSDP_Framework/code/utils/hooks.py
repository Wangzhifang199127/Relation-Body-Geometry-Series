import torch
from typing import Dict, Callable

class ActivationHook:
    """
    A context manager to extract activations (V, Router logits, Hidden States) from a model.
    """
    def __init__(self, model, target_layers: list, hook_type: str = "v"):
        self.model = model
        self.target_layers = target_layers
        self.hook_type = hook_type
        self.storage: Dict[int, torch.Tensor] = {}
        self.hooks = []

    def _get_hook_fn(self, layer_idx: int) -> Callable:
        def hook_fn(module, input, output):
            if self.hook_type == "v":
                # Assuming output of v_proj is direct tensor
                self.storage[layer_idx] = output.detach().float().cpu()
            elif self.hook_type == "router":
                # Handle tuple output for MoE Router
                logits = output[0] if isinstance(output, tuple) else output
                self.storage[layer_idx] = logits.detach().float().cpu()
        return hook_fn

    def __enter__(self):
        # Note: This is model-specific. Ensure you point to the correct attention/MLP layers.
        # Example for Qwen/Llama-like architectures:
        for idx in self.target_layers:
            if self.hook_type == "v":
                layer = self.model.model.layers[idx].self_attn.v_proj
            elif self.hook_type == "router":
                layer = self.model.model.layers[idx].mlp.gate
            
            self.hooks.append(layer.register_forward_hook(self._get_hook_fn(idx)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []