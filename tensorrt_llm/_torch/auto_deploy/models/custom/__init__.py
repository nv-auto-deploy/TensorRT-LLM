import tensorrt_llm._torch.auto_deploy.models.custom.new_sharding.modeling_deepseek_v2  # noqa: F401
import tensorrt_llm._torch.auto_deploy.models.custom.new_sharding.modeling_mistral  # noqa: F401

from .modeling_deepseek import DeepSeekV3ForCausalLM
from .modeling_deepseek_v2 import DeepSeekV2ForCausalLM
from .modeling_glm4_moe_lite import Glm4MoeLiteForCausalLM
from .modeling_kimi_k2 import KimiK2ForCausalLM, KimiK25ForConditionalGeneration
from .modeling_nemotron_flash import NemotronFlashForCausalLM, NemotronFlashPreTrainedTokenizerFast
from .new_sharding.modeling_deepseek import DeepSeekV3ForCausalLM  # noqa: F811
from .new_sharding.modeling_nemotron_h import NemotronHForCausalLM
from .new_sharding.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)

__all__ = (
    "DeepSeekV3ForCausalLM",
    "Glm4MoeLiteForCausalLM",
    "KimiK2ForCausalLM",
    "KimiK25ForConditionalGeneration",
    "NemotronFlashForCausalLM",
    "NemotronFlashPreTrainedTokenizerFast",
    "NemotronHForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
)
