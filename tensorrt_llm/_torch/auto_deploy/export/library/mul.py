"""Patch for torch.mul / a*b to use a simpler implementation during export."""

import torch

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("mul")
class MulPatch(BaseExportPatch):
    """Patch tensor-tensor multiply to a custom op; keep scalars native."""

    def _apply_patch(self):
        self.original_values["torch.mul"] = torch.mul
        self.original_values["torch.Tensor.mul"] = torch.Tensor.mul
        self.original_values["torch.Tensor.__mul__"] = torch.Tensor.__mul__
        self.original_values["torch.Tensor.__rmul__"] = torch.Tensor.__rmul__

        def _torch_mul_patch(a, b):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return torch.ops.auto_deploy.torch_mul_simple(a, b)
            return self.original_values["torch.mul"](a, b)

        def _tensor_mul_patch(self_tensor, other):
            if isinstance(other, torch.Tensor):
                return torch.ops.auto_deploy.torch_mul_simple(self_tensor, other)
            return self.original_values["torch.Tensor.mul"](self_tensor, other)

        def _tensor_dunder_mul_patch(self_tensor, other):
            if isinstance(other, torch.Tensor):
                return torch.ops.auto_deploy.torch_mul_simple(self_tensor, other)
            return self.original_values["torch.Tensor.__mul__"](self_tensor, other)

        def _tensor_dunder_rmul_patch(self_tensor, other):
            if isinstance(other, torch.Tensor):
                return torch.ops.auto_deploy.torch_mul_simple(other, self_tensor)
            return self.original_values["torch.Tensor.__rmul__"](self_tensor, other)

        torch.mul = _torch_mul_patch
        torch.Tensor.mul = _tensor_mul_patch
        torch.Tensor.__mul__ = _tensor_dunder_mul_patch
        torch.Tensor.__rmul__ = _tensor_dunder_rmul_patch

    def _revert_patch(self):
        torch.mul = self.original_values["torch.mul"]
        torch.Tensor.mul = self.original_values["torch.Tensor.mul"]
        torch.Tensor.__mul__ = self.original_values["torch.Tensor.__mul__"]
        torch.Tensor.__rmul__ = self.original_values["torch.Tensor.__rmul__"]
