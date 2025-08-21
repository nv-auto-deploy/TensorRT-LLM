"""Patch for torch.matmul to use a simpler implementation during export."""

import torch

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("matmul")
class MatmulPatch(BaseExportPatch):
    """Patch torch.matmul to use a simpler implementation for export.

    This patch replaces torch.matmul (and related tensor methods) with a version
    that calls a custom op to stabilize how the op appears in the exported graph.
    """

    def _apply_patch(self):
        """Apply the matmul patch."""
        # Store original functions
        self.original_values["torch.matmul"] = torch.matmul
        self.original_values["torch.Tensor.matmul"] = torch.Tensor.matmul
        self.original_values["torch.Tensor.__matmul__"] = torch.Tensor.__matmul__

        # Create patched functions
        def _torch_matmul_patch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.ops.auto_deploy.torch_matmul_simple(a, b)

        def _tensor_matmul_patch(self_tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.ops.auto_deploy.torch_matmul_simple(self_tensor, other)

        def _tensor_dunder_matmul_patch(
            self_tensor: torch.Tensor, other: torch.Tensor
        ) -> torch.Tensor:
            return torch.ops.auto_deploy.torch_matmul_simple(self_tensor, other)

        # Apply patches
        torch.matmul = _torch_matmul_patch
        torch.Tensor.matmul = _tensor_matmul_patch
        torch.Tensor.__matmul__ = _tensor_dunder_matmul_patch

    def _revert_patch(self):
        """Revert the matmul patch."""
        torch.matmul = self.original_values["torch.matmul"]
        torch.Tensor.matmul = self.original_values["torch.Tensor.matmul"]
        torch.Tensor.__matmul__ = self.original_values["torch.Tensor.__matmul__"]
