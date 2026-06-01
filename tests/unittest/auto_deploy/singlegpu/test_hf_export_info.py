# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator

import torch
from torch import nn
from torch.fx import symbolic_trace

from tensorrt_llm._torch.auto_deploy.models.hf import TextModelExportInfo


class _DummyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


def test_text_model_export_info_uses_scalar_embedding_keepalive_assert():
    model = _DummyTextModel()
    gm = symbolic_trace(model)

    export_info = TextModelExportInfo("dummy")
    export_info.post_process(model, gm)
    gm.recompile()
    gm.graph.lint()

    assert_node = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target == torch._assert
    )
    cond_node = assert_node.args[0]

    assert cond_node.op == "call_function"
    assert cond_node.target == operator.ge
    assert cond_node.args[0].op == "call_function"
    assert cond_node.args[0].target == torch.ops.aten.sym_size.int


class _NestedDelegatingTextModel(nn.Module):
    """Top-level model whose ``get_input_embeddings`` delegates to a child accessor.

    Mirrors ``NemotronHForCausalLM.get_input_embeddings``, which returns
    ``self.backbone.get_input_embeddings()`` instead of returning a submodule by
    direct attribute access.
    """

    def __init__(self):
        super().__init__()
        self.backbone = _DummyTextModel()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids)


def test_text_model_export_info_handles_nested_delegating_accessor():
    """Regression: the exposed accessor must fetch the submodule by path.

    ``expose_graph_module_accessor`` must not rebind the original accessor method
    onto the exported GraphModule. With nested delegation (NemotronH style), the
    intermediate container on the exported graph is a plain ``nn.Module`` without
    ``get_input_embeddings``, so rebinding the original method would raise
    ``AttributeError`` when it calls ``self.backbone.get_input_embeddings()``.
    """
    sub_mod = _NestedDelegatingTextModel()

    # Build a GraphModule the way export does: a plain ``nn.Module`` root with a
    # plain intermediate container (not the original ``_DummyTextModel`` class),
    # so the delegating accessor cannot be re-run on it.
    graph = torch.fx.Graph()
    placeholder = graph.placeholder("input_ids")
    graph.output(placeholder)
    sub_gm = torch.fx.GraphModule(nn.Module(), graph)
    sub_gm.add_module("backbone", nn.Module())

    TextModelExportInfo("dummy").post_process(sub_mod, sub_gm)
    sub_gm.recompile()
    sub_gm.graph.lint()

    exposed = sub_gm.get_input_embeddings()
    assert exposed is sub_gm.get_submodule("backbone.embed_tokens")
    assert exposed is sub_mod.get_input_embeddings()
    assert isinstance(exposed, nn.Embedding)
