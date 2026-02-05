
import argparse
import logging
import os
import time

import torch
import tensorrt as trt
import numpy as np
from diffusers import UNet2DModel

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark UNet: Torch vs TensorRT")
    parser.add_argument("--engine-path", type=str, default="unet_trt/unet.plan", help="Path to TensorRT engine")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 for benchmark (default: False, uses FP16)")
    return parser.parse_args()

def get_unet_model(device, dtype):
    # Same config as optimize_unet.py
    unet2d_config = {
        "sample_size": 32,
        "in_channels": 128,
        "out_channels": 128,
        "center_input_sample": False,
        "time_embedding_type": "positional",
        "freq_shift": 0,
        "flip_sin_to_cos": True,
        "down_block_types": ("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        "block_out_channels": [320, 640, 1280],
        "layers_per_block": 1,
        "mid_block_scale_factor": 1,
        "downsample_padding": 1,
        "downsample_type": "conv",
        "upsample_type": "conv",
        "dropout": 0.0,
        "act_fn": "silu",
        "norm_num_groups": 32,
        "norm_eps": 1e-05,
        "resnet_time_scale_shift": "default",
        "add_attention": False,
    }
    
    logger.info("Creating UNet2DModel...")
    unet = UNet2DModel(**unet2d_config).to(device, dtype=dtype)
    unet.requires_grad_(False)
    unet.eval()
    return unet

def benchmark_pipeline(name, func, warmth=10, runs=100):
    # Warmup
    for _ in range(warmth):
        with torch.no_grad():
            output = func()
    torch.cuda.synchronize()

    # Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(runs):
        with torch.no_grad():
            output = func()
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / runs
    fps = 1000.0 / avg_time_ms
    logger.info(f"{name}: {avg_time_ms:.2f} ms ({fps:.1f} FPS)")
    
    return avg_time_ms, fps, output

def benchmark_torch(unet, sample, timestep, warmth=10, runs=100):
    logger.info(f"Benchmarking Torch UNet ({sample.dtype})...")

    def run_pipeline():
        # UNet forward -> (sample,)
        return unet(sample, timestep, return_dict=False)[0]

    return benchmark_pipeline("Torch UNet", run_pipeline, warmth, runs)

def load_engine(engine_path):
    if not os.path.exists(engine_path):
        logger.warning(f"Engine not found: {engine_path}")
        return None
    logger.info(f"Loading engine from {engine_path}...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def benchmark_trt(engine_path, sample, timestep, warmth=10, runs=100):
    logger.info("Benchmarking TensorRT UNet...")

    engine = load_engine(engine_path)
    if not engine:
        return None, None, None

    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    
    # --- Prepare Buffers ---
    # Inputs: sample, timestep
    # Output: out_sample
    
    inputs = {}
    outputs = {}
    
    # Identify bindings (using name because order can vary, but usually sample=0, timestep=1)
    # We'll map by name for robustness if names were preserved, otherwise index.
    # optimize_unet.py export used input_names=["sample", "timestep"], output_names=["out_sample"]
    
    # B=1, C=128, H=32, W=32
    # Timestep B=1 (or scalar)
    
    # We must allocate output buffer
    out_shape = (1, 128, 32, 32)
    output_tensor = torch.zeros(out_shape, dtype=torch.float16, device="cuda")
    
    # Map inputs
    # Need to be careful with timestep dtype. In ONNX export we passed float.
    # If using FP16, inputs might expect FP16.
    
    # Helper to find tensor name by matching
    def find_tensor_name(target_name):
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if target_name in name: return name
        return None

    sample_name = find_tensor_name("sample")
    timestep_name = find_tensor_name("timestep")
    output_name = find_tensor_name("out_sample")
    
    if not sample_name or not output_name:
        # Fallback to indices if names were stripped (unlikely with TRT/ONNX)
        sample_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(2) # Assuming index 2? No, let's iterate.
        # This is risky without names. But our export script set names explicitly.
        
    # Set shapes
    context.set_input_shape(sample_name, sample.shape)
    if timestep_name:
         context.set_input_shape(timestep_name, timestep.shape)
    
    # Bind addresses
    context.set_tensor_address(sample_name, sample.data_ptr())
    if timestep_name:
        context.set_tensor_address(timestep_name, timestep.data_ptr())
    
    # Find output name if not found above by string match (it should be "out_sample")
    if not output_name:
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_name = name
                break
    
    context.set_tensor_address(output_name, output_tensor.data_ptr())

    def run_trt():
        context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        return output_tensor

    return benchmark_pipeline("TRT UNet", run_trt, warmth, runs)

def main():
    args = parse_args()

    use_fp16 = not args.fp32
    device = "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32

    # create dummy input
    # matches UNet config: 1, 128, 32, 32
    sample = torch.randn(1, 128, 32, 32, device=device, dtype=dtype)
    timestep = torch.tensor([1.0], device=device, dtype=dtype)

    # 1. Torch
    unet = get_unet_model(device, dtype)
    torch_ms, torch_fps, _ = benchmark_torch(unet, sample, timestep)
    
    # Free memory
    del unet
    torch.cuda.empty_cache()
    
    print("-" * 50)

    # 2. TensorRT
    trt_ms, trt_fps, _ = benchmark_trt(args.engine_path, sample, timestep)
    
    if trt_ms:
        print("\n" + "=" * 50)
        print("   UNET BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"Torch:      {torch_ms:6.2f} ms  ({torch_fps:6.1f} FPS)")
        print(f"TensorRT:   {trt_ms:6.2f} ms  ({trt_fps:6.1f} FPS)")
        print("-" * 50)
        print(f"Speedup:    {torch_ms/trt_ms:6.2f}x")
        print("=" * 50 + "\n")
    else:
        print("TensorRT benchmark failed or engine not found.")

if __name__ == "__main__":
    main()
