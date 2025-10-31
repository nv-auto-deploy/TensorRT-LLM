import argparse
from collections import abc

import modelopt.torch.opt as mto
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm

from tensorrt_llm._torch.auto_deploy.compile import CompileBackendRegistry
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils._graph import load_buffers_and_params
from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger

torch._dynamo.config.cache_size_limit = 100

dtype_map = {
    "Half": torch.float16,
    "BFloat16": torch.bfloat16,
    "Float": torch.float32,
}


# TODO: Reuse the cache context from the original model
class TransformerWrapper(torch.nn.Module):
    def __init__(self, compiled_model, config):
        super().__init__()
        self.model = compiled_model
        self.config = config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def cache_context(self, *args, **kwargs):
        # Return a no-op context manager since the compiled model
        # doesn't support this feature
        from contextlib import contextmanager

        @contextmanager
        def noop_context():
            yield

        return noop_context()


@torch.inference_mode()
def generate_image(pipe: DiffusionPipeline, prompt: str, image_name: str) -> None:
    """Generate an image using the given pipeline and prompt."""
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=30,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    image.save(image_name)
    ad_logger.info(f"Image generated saved as {image_name}")


@torch.inference_mode()
def benchmark_backbone_standalone(
    pipe, num_warmup=10, num_benchmark=100, model_name="flux-dev", model_dtype="Half"
):
    """Benchmark the backbone model directly without running the full pipeline."""
    backbone = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet

    # Generate dummy inputs for the backbone
    dummy_inputs = _gen_dummy_inp_flux(backbone)

    # Warmup
    print(f"Warming up: {num_warmup} iterations")
    for _ in tqdm(range(num_warmup), desc="Warmup"):
        _ = backbone(**dummy_inputs)

    # Benchmark
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Benchmarking: {num_benchmark} iterations")
    times = []
    for _ in tqdm(range(num_benchmark), desc="Benchmark"):
        start_event.record()
        _ = backbone(**dummy_inputs)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    avg_latency = sum(times) / len(times)
    times = sorted(times)
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    p99 = times[int(len(times) * 0.99)]

    print(f"\nBackbone-only inference latency ({model_dtype}):")
    print(f"  Average: {avg_latency:.2f} ms")
    print(f"  P50: {p50:.2f} ms")
    print(f"  P95: {p95:.2f} ms")
    print(f"  P99: {p99:.2f} ms")

    return avg_latency


def torch_to(data, *args, **kwargs):
    """Try to recursively move the data to the specified args/kwargs."""
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, (tuple, list)):
        return type(data)([torch_to(val, *args, **kwargs) for val in data])
    elif isinstance(data, abc.Mapping):
        return {k: torch_to(val, *args, **kwargs) for k, val in data.items()}
    return data


def _gen_dummy_inp_flux(backbone, min_bs=1):
    cfg = backbone.config
    text_maxlen = 512
    img_dim = 4096

    dtype = torch.bfloat16
    dummy_input = {
        "hidden_states": torch.randn(min_bs, img_dim, cfg.in_channels, dtype=dtype),
        "encoder_hidden_states": torch.randn(
            min_bs, text_maxlen, cfg.joint_attention_dim, dtype=dtype
        ),
        "pooled_projections": torch.randn(min_bs, cfg.pooled_projection_dim, dtype=dtype),
        "timestep": torch.ones(1, dtype=dtype),
        "img_ids": torch.randn(img_dim, 3, dtype=torch.float32),
        "txt_ids": torch.randn(text_maxlen, 3, dtype=torch.float32),
        "return_dict": False,
        "joint_attention_kwargs": {},
    }
    if cfg.guidance_embeds:  # flux-dev
        dummy_input["guidance"] = torch.full((1,), 3.5, dtype=torch.float32)

    dummy_input = torch_to(dummy_input, device="cuda")

    return dummy_input


def execution_device_getter(self):
    return torch.device("cuda")


def execution_device_setter(self, value):
    self.__dict__["_execution_device"] = torch.device("cuda")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="The model to use for inference.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of an astronaut riding a horse on mars",
        help="The prompt to use for inference.",
    )
    parser.add_argument(
        "--hf_inference",
        action="store_true",
        help="Whether to generate image with the base hf model in addition to autodeploy generation",
    )
    parser.add_argument(
        "--restore_from", type=str, help="The quantized checkpoint path to restore the model from"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Whether to benchmark the model",
    )
    parser.add_argument(
        "--skip_image_generation",
        action="store_true",
        help="Whether to skip image generation",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="The max batch size to use for the model",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch-opt",
        help="The backend to use for compilation (default: torch-opt)",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="output.png",
        help="Path to save the generated image (default: output.png)",
    )
    args = parser.parse_args()

    # Validate max_batch_size
    if args.max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive, got {args.max_batch_size}")

    DiffusionPipeline._execution_device = property(execution_device_getter, execution_device_setter)
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    if args.hf_inference:
        if not args.skip_image_generation:
            ad_logger.info("Generating image with the torch pipeline")
            hf_image_path = f"hf_{args.image_path}"
            generate_image(pipe, args.prompt, hf_image_path)
        if args.benchmark:
            ad_logger.info("Benchmarking HuggingFace model")
            latency = benchmark_backbone_standalone(pipe, model_dtype="BFloat16")
            ad_logger.info(f"HuggingFace Average Inference Latency: {latency:.2f} ms")
    model = pipe.transformer
    flux_config = pipe.transformer.config
    flux_kwargs = _gen_dummy_inp_flux(model, min_bs=args.max_batch_size)

    gm = torch_export_to_gm(model, args=(), kwargs=flux_kwargs, clone=False)

    if args.restore_from:
        ad_logger.info(f"Restoring model from {args.restore_from}")
        try:
            mto.restore(model, args.restore_from)
            quant_state_dict = model.state_dict()
            load_buffers_and_params(
                gm, quant_state_dict, strict_missing=False, strict_unexpected=False, clone=False
            )
        except Exception as e:
            ad_logger.error(f"Failed to restore model from {args.restore_from}: {e}")
            raise

    # Apply inference optimizer fusions
    ad_logger.info("Applying inference optimizer fusions (FP8 and FP4)...")
    optimizer_config = {
        "fuse_fp8_linear": {
            "stage": "post_load_fusion",
            "backend": "torch",
        },
        "fuse_nvfp4_linear": {
            "stage": "post_load_fusion",
            "backend": "trtllm",
        },
    }
    optimizer = InferenceOptimizer(factory=None, config=optimizer_config)
    gm = optimizer(cm=None, mod=gm)
    ad_logger.info("Inference optimizer fusions applied successfully")

    # Validate backend availability
    if not CompileBackendRegistry.has(args.backend):
        available = CompileBackendRegistry.list()
        raise ValueError(f"Backend '{args.backend}' not found. Available backends: {available}")

    compiler_cls = CompileBackendRegistry.get(args.backend)
    gm = compiler_cls(gm, args=(), max_batch_size=args.max_batch_size, kwargs=flux_kwargs).compile()

    del model
    fx_model = TransformerWrapper(gm, flux_config)
    pipe.transformer = fx_model
    if not args.skip_image_generation:
        ad_logger.info("Generating image with the exported auto-deploy model")
        generate_image(pipe, args.prompt, args.image_path)

    if args.benchmark:
        ad_logger.info("Benchmarking AutoDeploy model")
        latency = benchmark_backbone_standalone(pipe, model_dtype="BFloat16")
        ad_logger.info(f"AutoDeploy Average Inference Latency: {latency:.2f} ms")


if __name__ == "__main__":
    main()
