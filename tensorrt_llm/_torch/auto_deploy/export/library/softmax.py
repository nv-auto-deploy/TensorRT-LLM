"""Patch for torch.nn.functional.softmax to use a simpler implementation during export."""

from typing import Optional

import torch
import torch.nn.functional as F

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("softmax")
class SoftmaxPatch(BaseExportPatch):
    """Patch F.softmax to use a custom op for export stability."""

    def _apply_patch(self):
        self.original_values["F.softmax"] = F.softmax
        self.original_values["torch.softmax"] = torch.softmax
        self.original_values["torch.Tensor.softmax"] = torch.Tensor.softmax

        def _f_softmax_patch(input: torch.Tensor, dim: Optional[int] = None, dtype=None):
            if dim is None:
                dim = -1
            return torch.ops.auto_deploy.torch_softmax_simple(input, dim, dtype)

        F.softmax = _f_softmax_patch

        def _torch_softmax_patch(input: torch.Tensor, dim: int = -1, dtype=None):
            return torch.ops.auto_deploy.torch_softmax_simple(input, dim, dtype)

        torch.softmax = _torch_softmax_patch

        def _tensor_softmax_patch(self_tensor: torch.Tensor, dim: int = -1, dtype=None):
            return torch.ops.auto_deploy.torch_softmax_simple(self_tensor, dim, dtype)

        torch.Tensor.softmax = _tensor_softmax_patch

    def _revert_patch(self):
        F.softmax = self.original_values["F.softmax"]
        torch.softmax = self.original_values["torch.softmax"]
        torch.Tensor.softmax = self.original_values["torch.Tensor.softmax"]
