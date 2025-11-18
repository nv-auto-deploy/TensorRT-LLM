from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

stream_dict = {"linear_aux_stream": torch.cuda.Stream()}
event_dict = {
    "aux_stream_event": torch.cuda.Event(),
    "main_stream_event": torch.cuda.Event(),
}


@torch.library.custom_op("auto_deploy::multi_stream_linear", mutates_args=())
def multi_stream_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    output = torch.ops.aten.linear(input, weight, bias)
    return output


@multi_stream_linear.register_fake
def multi_stream_linear_fake(input, weight, bias):
    """Fake implementation of multi_stream_linear."""
    return torch.ops.aten.linear(input, weight, bias)


def aux_stream_wrapper(
    fn: Callable,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Callable:
    stream_name = kwargs.pop("stream_name")
    event_dict["main_stream_event"].record()
    with torch.cuda.stream(stream_dict[stream_name]):
        event_dict["main_stream_event"].wait()
        output = fn(*args)
        event_dict["aux_stream_event"].record()
    event_dict["aux_stream_event"].wait()
    return output


def replace_multi_stream_linear_with_aux_stream_wrapper(
    gm: GraphModule, aux_stream_wrapper: Callable[..., Any], stream_name: str
) -> Tuple[GraphModule, int]:
    """Traverse ``gm`` and replace all ``auto_deploy::multi_stream_linear`` ops with ``aux_stream_wrapper``.

    The replacement preserves the original args/kwargs of the node.
    After rewriting, the graph is cleaned and recompiled.

    Args:
        gm: The FX graph module to transform.
        aux_stream_wrapper: A callable to replace the custom op with.

    Returns:
        A tuple of (gm, num_replaced)
    """
    graph = gm.graph
    num_replaced = 0

    # Collect targets first to avoid mutating while iterating
    target_nodes: list[Node] = []
    for n in graph.nodes:
        if is_op(n, torch.ops.auto_deploy.multi_stream_linear):
            target_nodes.append(n)

    for n in target_nodes:
        with graph.inserting_after(n):
            kwargs = n.kwargs.copy()
            kwargs["stream_name"] = stream_name
            new_node = graph.call_function(
                aux_stream_wrapper, args=(n.target, *n.args), kwargs=kwargs
            )
        n.replace_all_uses_with(new_node)
        graph.erase_node(n)
        num_replaced += 1

    if num_replaced:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return gm, num_replaced


class ParallelTwoLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return multi_stream_linear(x, self.fc1.weight, self.fc1.bias) + self.fc2(x)


in_dim, out_dim = 128, 256
aux_stream = torch.cuda.Stream()
event0 = torch.cuda.Event()
event1 = torch.cuda.Event()

model = ParallelTwoLinear(in_dim, out_dim).eval().to("cuda")

# Example input used for export
example_input = torch.randn(4, in_dim).to("cuda")

# Export the graph
egm = torch.export.export(model, (example_input,))
gm = egm.module()
output = gm(example_input)

test_x = torch.randn(4, in_dim).to("cuda")
ref_output = model(test_x)

# pattern matching and replace
gm, num_replaced = replace_multi_stream_linear_with_aux_stream_wrapper(
    gm, aux_stream_wrapper, "linear_aux_stream"
)
print(f"Replaced {num_replaced} nodes")
print(gm.graph)
y = gm(test_x)
assert torch.allclose(y, ref_output)

static_x = torch.randn(4, in_dim).to("cuda")
static_output = torch.randn(4, out_dim).to("cuda")

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_output.copy_(gm(static_x))

static_x.copy_(test_x)
graph.replay()

assert torch.allclose(static_output, ref_output)

for i in range(100):
    graph.replay()
