from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

stream_dict = {"aux_stream": torch.cuda.Stream()}
event_dict = {
    "aux_stream_event": torch.cuda.Event(),
    "main_stream_event": torch.cuda.Event(),
}


@torch.library.custom_op("auto_deploy::record_event", mutates_args=())
def record_event(event_name: str) -> None:
    event = event_dict[event_name]
    event.record()


@torch.library.custom_op("auto_deploy::wait_event", mutates_args=())
def wait_event(event_name: str) -> None:
    event = event_dict[event_name]
    event.wait()


@torch.library.custom_op("auto_deploy::multi_stream_linear", mutates_args=())
def multi_stream_linear(
    input: torch.Tensor, weight0: torch.Tensor, weight1: torch.Tensor
) -> torch.Tensor:
    output = torch.ops.aten.linear(input, weight0)
    output = torch.ops.aten.linear(output, weight1)
    return output


@multi_stream_linear.register_fake
def multi_stream_linear_fake(input, weight0, weight1):
    """Fake implementation of multi_stream_linear."""
    output = torch.ops.aten.linear(input, weight0)
    return torch.ops.aten.linear(output, weight1)


def node_wrapper_with_record_event(
    fn: Callable,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Callable:
    torch.ops.auto_deploy.record_event("main_stream_event")
    output = fn(*args, **kwargs)
    return output


def aux_stream_wrapper(
    fn: Callable,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Callable:
    stream_name = kwargs.pop("stream_name")
    with torch.cuda.stream(stream_dict[stream_name]):
        torch.ops.auto_deploy.wait_event("main_stream_event")
        output = fn(*args)
        torch.ops.auto_deploy.record_event("aux_stream_event")
    torch.ops.auto_deploy.wait_event("aux_stream_event")
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
        target_input_node = None
        for input_node in n.all_input_nodes:
            if len(input_node.users) > 1:
                target_input_node = input_node
                break
        if target_input_node is None:
            raise ValueError(f"Target input node not found for node {n}")
        with graph.inserting_before(target_input_node):
            new_node = graph.call_function(
                node_wrapper_with_record_event,
                args=(target_input_node.target, *target_input_node.args),
                kwargs=target_input_node.kwargs,
            )
            target_input_node.replace_all_uses_with(new_node)
            graph.erase_node(target_input_node)
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
        self.fc10 = nn.Linear(in_dim, in_dim)
        self.fc11 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(x)
        y0 = self.fc2(x)
        y1 = multi_stream_linear(x, self.fc10.weight, self.fc11.weight)
        return y0 + y1


in_dim, out_dim = 128, 256

model = (
    nn.Sequential(ParallelTwoLinear(in_dim, out_dim), ParallelTwoLinear(out_dim, out_dim))
    .eval()
    .to("cuda")
)

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
    gm, aux_stream_wrapper, "aux_stream"
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
    gm(torch.randn(4, in_dim).to("cuda"))

for i in range(100):
    graph.replay()
