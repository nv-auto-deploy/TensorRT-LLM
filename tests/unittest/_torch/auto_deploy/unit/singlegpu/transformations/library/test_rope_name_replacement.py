import pytest

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.transform.library.rope_preexport import (
    NoopRotaryWrapper,
    replace_rope_modules_by_name,
)


def test_replace_rope_modules_by_name_glm47_rotary_emb_meta():
    transformers = pytest.importorskip("transformers")
    accelerate = pytest.importorskip("accelerate")

    AutoConfig = transformers.AutoConfig
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    init_empty_weights = accelerate.init_empty_weights

    model_id = "zai-org/GLM-4.7-FP8"

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:  # network/cache dependent
        pytest.skip(f"Could not load HF config for {model_id}: {e}")

    # Instantiate on meta (no weights)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    res = replace_rope_modules_by_name(m)

    # For GLM-4.7-FP8, this is the rotary embedding module.
    assert "model.rotary_emb" in res.replaced
    assert isinstance(m.model.rotary_emb, NoopRotaryWrapper)
    assert m.model.rotary_emb.inner.__class__.__name__.lower().endswith("rotaryembedding")
