import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm, Qwen3MoeSparseMoeBlock

from tensorrt_llm._torch.auto_deploy.shim import AutoDeployConfig
from tensorrt_llm._torch.auto_deploy.transformations.library import sharding
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams


class LayerwiseNvtxMarker(object):
    """This module contains all the code needed to enable forward hooks in a pytorch network.

    To register the hooks for a given network, the user needs to instantiate a LayerwiseNvtxMarker object.
    Then call the register_hooks method.

    Example:
        my_layerwise_nvtx_marker = LayerwiseNvtxMarker()
        my_layerwise_nvtx_marker.register_hooks(my_network_model)
    """

    def __init__(self):
        """Initialize module variables.

        Args:
            None:

        Returns:
            None:

        Raises:
            None:
        """
        super().__init__()
        self.module_to_name_map = {}
        self.out_tensor_to_name_stack = {}
        self.iteration = 0

    @staticmethod
    def _print_tensor(tensor_obj, prefix, tensor_list=[]):
        """Descends iterators that contains Tensors and prints the Tensor.

        Recursive function that descends iterator type arguments until
        it finds a Tensor object.

        Args:
            tensor_obj: Could be a Tensor or an iterator type that contains Tensors
            prefix: String name to assign to the Tensor

        Returns:
            None:

        Raises:
            None:
        """
        tensor_dims = []
        if isinstance(tensor_obj, list) or isinstance(tensor_obj, tuple):
            for ten in tensor_obj:
                tensor_list = LayerwiseNvtxMarker._print_tensor(ten, prefix, tensor_list)
        elif isinstance(tensor_obj, torch.Tensor):
            hex(id(tensor_obj))
            tensor_dims = list(tensor_obj.size())
            tensor_list.append(tensor_dims)
        return tensor_list

    def _module_fwd_hook(self, module_obj, in_tensor, out_tensor):
        """Callback function that ends the NVTX marker.

        Records the module name and tensor information
        Called after the module executes the forward method.

        Args:
            module_obj: Pointer to the module object
            in_tensor: Input tensor or list of tensors
            out_tensor: Output tensor of the resulting forward operator

        Returns:
            None:

        Raises:
            None:
        """
        nvtx.range_pop()
        module_name = self.module_to_name_map[module_obj]

        logger.debug(f"FWD hook module {module_name}")
        if module_name == "'top'":
            self.iteration = self.iteration + 1
            logger.debug(f"Completed {self.iteration} iterations")

        return

    def _module_fwd_pre_hook(self, module_obj, in_tensor):
        """Creates an NVTX marker with the module name in it. This function is called before the module executes.

        Args:
            module_obj: Module object data structure - used to get unique module name
            in_tensor: Input tensor data structure

        Returns:
            None

        Raises:
            None
        """
        marker_dict = {}
        module_name = self.module_to_name_map[module_obj]
        module_params = module_obj.named_parameters(recurse=False)
        logger.debug(f"FWD Pre hook module:{module_name}")
        marker_dict["Module"] = module_name
        marker_dict["TrainableParams"] = {}
        ## Get trainable parameters like weights and bias
        for param_name, param_obj in module_params:
            marker_dict["TrainableParams"][param_name] = list(param_obj.size())
            logger.debug(f"Param {param_name} value {list(param_obj.size())}")

        in_tensor_list = LayerwiseNvtxMarker._print_tensor(in_tensor, "Input", tensor_list=[])
        if in_tensor_list:
            marker_dict["Inputs"] = in_tensor_list
            logger.debug("Input Tensor List-> {in_tensor_list}")

        nvtx.range_push("{}".format(marker_dict))

        return

    def register_hooks(self, network_model, module_prefix="top"):
        """User level function that activates all the hooks.

        The user needs to call this method from the network source code
        The code descends all the modules in the network and registers their
        respective hooks.

        Args:
            network_model: Model object for the network
            module_prefix: (default: top)

        Returns:
            None

        Raises:
            Exception if a module instance is reused
        """
        for name, module in network_model.named_modules(prefix=module_prefix):
            logger.debug(f"Module Name:{name} addr:{hex(id(module))}")
            module.register_forward_pre_hook(self._module_fwd_pre_hook)
            module.register_forward_hook(self._module_fwd_hook)
            if module not in self.module_to_name_map:
                self.module_to_name_map[module] = name
            else:
                raise Exception("Module instance {} is not unique ".format(module))
        return


def _forward_rmsnorm(self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor):
    return torch.ops.trtllm.fused_rmsnorm(hidden_states, self.weight, self.bias, self.eps)


# patch for MoE to reduce torch.export time
def _forward_moe(self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    # print(f"batch_size: {batch_size}, sequence_length: {sequence_length}, hidden_dim: {hidden_dim}")
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.ops.moe.torch_moe(
        hidden_states,
        selected_experts,
        routing_weights,
        w1_weight=[expert.gate_proj.weight for expert in self.experts],
        w2_weight=[expert.down_proj.weight for expert in self.experts],
        w3_weight=[expert.up_proj.weight for expert in self.experts],
    )
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


Qwen3MoeSparseMoeBlock._original_forward = Qwen3MoeSparseMoeBlock.forward
Qwen3MoeSparseMoeBlock.forward = _forward_moe

if False:
    Qwen3MoeRMSNorm._original_forward = Qwen3MoeRMSNorm.forward
    Qwen3MoeRMSNorm.forward = _forward_rmsnorm


# fix sharder bug
def _insert_sharded_matmul_patch(gm, n, dim, rank, world_size, add_dist=False):
    _insert_sharded_matmul_original(gm, n, 0, rank, world_size, add_dist=True)


_insert_sharded_matmul_original = sharding._insert_sharded_matmul
sharding._insert_sharded_matmul = _insert_sharded_matmul_patch


layerwise_nvtx_marker = LayerwiseNvtxMarker()
module_prefix = "Model"
try:
    layerwise_nvtx_marker.register_hooks(transformers.models.qwen3_moe, module_prefix)
except Exception as e:
    print(f"Error registering hooks: {e}")


def main():
    model_name = "Qwen/Qwen3-235B-A22B"  # Qwen/Qwen3-30B-A3B, Qwen/Qwen3-8B, Qwen/Qwen3-235B-A22B

    # load the tokenizer and the model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    print(model)

    # Build Config
    build_config = BuildConfig(max_seq_len=2048, max_batch_size=4)

    # Configure AutoDeploy
    ad_config = AutoDeployConfig(
        use_cuda_graph=False,
        torch_compile_enabled=False,
        attn_backend="FlashInfer",
        model_kwargs={"num_hidden_layers": 4},
    )

    # Create the LLM object
    llm = LLM(
        model=model_name,
        backend="autodeploy",
        build_config=build_config,
        pytorch_backend_config=ad_config,
        tensor_parallel_size=4,
        verbose=True,
    )
    torch.cuda.cudart().cudaProfilerStart()
    # Generate text
    with torch.inference_mode():
        output = llm.generate(
            "Write a short poem about AI",
            sampling_params=SamplingParams(
                max_tokens=100,
                top_k=50,
                temperature=0.7,
            ),
        )
        print(f"output: {output.outputs[0].text}, size: {len(output.outputs[0].text)}")
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
