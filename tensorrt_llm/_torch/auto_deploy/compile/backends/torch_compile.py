"""Backend that uses torch.compile only."""

import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger

from ..compiler import BackendCompiler, BackendRegistry


@BackendRegistry.register("torch-compile")
class TorchCompileCompiler(BackendCompiler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_graph_batch_sizes = self.compiler_kwargs.get("cuda_graph_batch_sizes")
        if not self.cuda_graph_batch_sizes:
            self.cuda_graph_batch_sizes = self._get_graph_batch_sizes(self.max_batch_size)
        ad_logger.info(f"Setting cuda_graph_batch_sizes to {self.cuda_graph_batch_sizes}")

        torch._dynamo.config.recompile_limit = max(
            len(self.cuda_graph_batch_sizes), torch._dynamo.config.recompile_limit
        )
        ad_logger.info(f"Setting recompile limit to {torch._dynamo.config.recompile_limit}")

        # Global torch config, set the torch compile cache to fix up to llama 405B
        torch._dynamo.config.cache_size_limit = 20
        ad_logger.info(f"Setting cache size limit to {torch._dynamo.config.cache_size_limit}")

    def compile(self) -> nn.Module:
        """Compile the model using torch.compile."""
        return torch.compile(self.gm, dynamic=True)
