import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLayer:
    def __init__(self, disable_adapter, **kwargs):
        super().__init__()
        self.k = {}
        self._sparse_alpha = {}
        self._sparse_dropout = nn.ModuleDict({})
        self._sparse_weight = nn.ParameterDict({})
        self._sparse_mask = nn.ParameterDict({})
        self.disable_adapter = disable_adapter
        self.kwargs = kwargs

    def update_layer(
        self, adapter_name, sparse_mask, sparse_alpha, sparse_dropout,
        inplace=False
    ):
        self._sparse_alpha[adapter_name] = sparse_alpha
        if sparse_dropout > 0.0:
            dropout = nn.Dropout(p=sparse_dropout)
        else:
            dropout = nn.Identity()
        self._sparse_dropout[adapter_name] = dropout
        # Actual trainable parameters
        sparse_weights = torch.zeros(
            sparse_mask.nelement(),
            dtype=self.weight.dtype, device=self.weight.device)
        if inplace:
            self._sparse_weight[adapter_name].data = sparse_weights
            self._sparse_mask[adapter_name].data = sparse_mask
        else:
            self._sparse_weight[adapter_name] = nn.Parameter(
                sparse_weights, requires_grad=True)
            self._sparse_mask[adapter_name] = nn.Parameter(
                sparse_mask, requires_grad=False)
        self._sparse_alpha[adapter_name] = sparse_alpha
        self.to(self.weight.device)

    def reset_sparse_parameters(self, adapter_name):
        if adapter_name in self._sparse_weight:
            nn.init.zeros_(self._sparse_weight[adapter_name]['weight'])

class LinearSparse(nn.Linear, SparseLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparse_method: str = 'random',
        sparse_alpha: float = 1.0,
        sparse_dropout: float = 0.0,
        adapter_name: str = 'sparse_adapter',
        disable_adapter: bool = False,
        **kwargs,
    ):
        self.sparse_method = sparse_method
        self.sparse_alpha = sparse_alpha
        self.sparse_dropout = float(sparse_dropout)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SparseLayer.__init__(self, disable_adapter, **kwargs)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        # self.zero_tensor = torch.zeros(size=self.weight.shape).flatten()
        self.unflattened_size = tuple(self.weight.shape)
        nn.Linear.reset_parameters(self)
        self.active_adapter = adapter_name

    def use_original_linear(self, use):
        self.weight.requires_grad = use
        self.disable_adapter = use

    def apply_adapter_weight(self):
        if self.active_adapter not in self._sparse_weight.keys():
            return
        sparse_weight = self._sparse_weight[self.active_adapter]
        mask = self._sparse_mask[self.active_adapter]
        scaling = self._sparse_alpha[self.active_adapter]
        scaled_output = sparse_weight * scaling
        new_weight = torch.scatter_add(
            self.weight.type(scaled_output.dtype).flatten(), dim=0, index=mask, src=scaled_output)
        new_weight = new_weight.view(self.unflattened_size)
        self.weight.data = new_weight
        sparse_weight.data = torch.zeros_like(sparse_weight)

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, bias=self.bias)

    def update_weight_selection_by_topk_score(self, adapter_name, score, k):
        if 0 <= k < 1:
            k = math.ceil(k * self.weight.nelement())
        elif k >= 1:
            k = int(k)
        else:
            raise ValueError(f"Invalid k number: {k}. k must be > 0")
        _, mask = torch.topk(score.flatten(), k, sorted=True)
        self.update_layer(
            adapter_name, mask, self.sparse_alpha, self.sparse_dropout)

    def update_weight_selection_by_mask(self, adapter_name, mask):
        self.update_layer(
            adapter_name, mask, self.sparse_alpha, self.sparse_dropout)

    def forward(self, x):
        previous_dtype = x.dtype

        if self.active_adapter not in self._sparse_weight.keys():
            return self._linear(x)

        if self.disable_adapter or self.active_adapter is None:
            return self._linear(x)

        sparse_weight = self._sparse_weight[self.active_adapter]
        mask = self._sparse_mask[self.active_adapter]
        dropout = self._sparse_dropout[self.active_adapter]
        scaling = self._sparse_alpha[self.active_adapter]
        # Apply sparse adapter
        scaled_output = sparse_weight * scaling
        # Scatter adapted values into weight tensor
        new_weight = torch.scatter_add(
            self.weight.type(scaled_output.dtype).flatten(), dim=0, index=mask, src=scaled_output)
        new_weight = new_weight.view(self.unflattened_size)
        x = x.to(sparse_weight.dtype)
        result = F.linear(dropout(x), new_weight, bias=self.bias)
        return result.to(previous_dtype)

    def extract_sparse_params(self):
        return self.sparse_params[self.active_adapter].state_dict()