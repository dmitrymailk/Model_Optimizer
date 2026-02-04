
import argparse
import logging
import os
import time
import sys

import torch
import tensorrt as trt
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, AutoModel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Flux VAE: Torch vs TensorRT")
    parser.add_argument("--model-path", type=str, default="fal/FLUX.2-Tiny-AutoEncoder", help="HuggingFace model ID or local path")
    parser.add_argument("--decoder-path", type=str, default="flux_vae_tiny_trt/vae_decoder.plan", help="Path to TensorRT Decoder engine")
    parser.add_argument("--encoder-path", type=str, default="flux_vae_tiny_trt/vae_encoder.plan", help="Path to TensorRT Encoder engine")
    parser.add_argument("--float32", action="store_true", help="Use FP32 for benchmark (default: False, uses FP16)")
    parser.add_argument("--image-path", type=str, default=r"C:\programming\auto_remaster\inference_optimization\170_2x.png", help="Path to input image for benchmarking")
    return parser.parse_args()


def process_image(image_path, size=512):
    image = Image.open(image_path).convert("RGB")
    
    # Resize ensuring shortest edge is 'size', then center crop
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # Map to [-1, 1]
    ])
    
    pixel_values = transform(image)
    pixel_values = pixel_values.unsqueeze(0) # Add batch dim -> (1, 3, H, W)
    return pixel_values

def benchmark_component(name, func, warmth=10, runs=100):
    # Warmup
    for _ in range(warmth):
        with torch.no_grad():
            func()
    torch.cuda.synchronize()
    
    # Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(runs):
        with torch.no_grad():
            func()
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / runs
    fps = 1000.0 / avg_time_ms
    logger.info(f"{name}: {avg_time_ms:.2f} ms ({fps:.1f} FPS)")
    return avg_time_ms

def benchmark_torch(vae, pixel_values, warmth=10, runs=100):
    logger.info(f"Benchmarking Torch Pipeline ({pixel_values.dtype})...")
    
    # 1. Measure Encoder
    def run_encoder():
        encoded_output = vae.encode(pixel_values)
        if hasattr(encoded_output, "latent_dist"):
            latents = encoded_output.latent_dist.sample()
        elif hasattr(encoded_output, "latents"):
            latents = encoded_output.latents
        else:
            latents = encoded_output[0]
        if hasattr(vae.config, "scaling_factor") and vae.config.scaling_factor is not None:
            latents = latents * vae.config.scaling_factor
        return latents

    encoder_ms = benchmark_component("Torch Encoder", run_encoder, warmth, runs)
    
    # Get latents for decoder benchmark
    latents = run_encoder()
    
    # 2. Measure Decoder
    def run_decoder():
        _ = vae.decode(latents, return_dict=False)[0]

    decoder_ms = benchmark_component("Torch Decoder", run_decoder, warmth, runs)
    
    # 3. Full Pipeline
    def run_pipeline():
        l = run_encoder()
        _ = vae.decode(l, return_dict=False)[0]
        
    pipeline_ms = benchmark_component("Torch Full Pipeline", run_pipeline, warmth, runs)
    
    return pipeline_ms, encoder_ms, decoder_ms

def load_engine(engine_path):
    if not os.path.exists(engine_path):
        logger.warning(f"Engine not found: {engine_path}")
        return None
    logger.info(f"Loading engine from {engine_path}...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def benchmark_trt(encoder_path, decoder_path, vae, pixel_values, warmth=10, runs=100):
    logger.info("Benchmarking TensorRT Pipeline...")
    
    enc_engine = load_engine(encoder_path)
    dec_engine = load_engine(decoder_path)
    
    if not dec_engine:
        logger.error("Decoder engine required for TRT benchmark.")
        return None, None
        
    dec_context = dec_engine.create_execution_context()
    enc_context = enc_engine.create_execution_context() if enc_engine else None
    
    stream = torch.cuda.Stream()
    
    # --- Prepare Data for Decoder Benchmark (Hybrid) ---
    # We still need this to benchmark Decoder in isolation using Torch Latents
    with torch.no_grad():
        encoded_output_torch = vae.encode(pixel_values)
        if hasattr(encoded_output_torch, "latent_dist"):
             latents_torch = encoded_output_torch.latent_dist.sample()
        else:
             latents_torch = encoded_output_torch.latents if hasattr(encoded_output_torch, "latents") else encoded_output_torch[0]
        if hasattr(vae.config, "scaling_factor") and vae.config.scaling_factor is not None:
             latents_torch = latents_torch * vae.config.scaling_factor

    # --- Setup Decoder (Common) ---
    dec_input_name = "latent"
    dec_output_name = "image"
    for i in range(dec_engine.num_io_tensors):
        name = dec_engine.get_tensor_name(i)
        if dec_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT: dec_input_name = name
        elif dec_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT: dec_output_name = name
    
    dec_context.set_input_shape(dec_input_name, latents_torch.shape)
    
    # Decoder Output Buffer
    output_shape = (latents_torch.shape[0], 3, latents_torch.shape[2]*8, latents_torch.shape[3]*8)
    image_out_tensor = torch.zeros(output_shape, dtype=torch.float16, device="cuda")
    dec_context.set_tensor_address(dec_output_name, image_out_tensor.data_ptr())

    # --- Benchmark TRT Decoder (Isolated) ---
    dec_context.set_tensor_address(dec_input_name, latents_torch.data_ptr())
    
    def run_decoder_trt():
        dec_context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

    decoder_ms = benchmark_component("TRT Decoder (Isolated)", run_decoder_trt, warmth, runs)
    
    encoder_ms = None
    full_ms = None
    
    # --- Benchmark TRT Encoder (If available) ---
    if enc_context:
        enc_input_name = "image"
        enc_output_name = "latent"
        for i in range(enc_engine.num_io_tensors):
            name = enc_engine.get_tensor_name(i)
            if enc_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT: enc_input_name = name
            elif enc_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT: enc_output_name = name
            
        enc_context.set_input_shape(enc_input_name, pixel_values.shape)
        
        # Encoder Output Buffer (Latents)
        # We know latents shape from torch check or infer
        latents_trt = torch.zeros_like(latents_torch)
        
        enc_context.set_tensor_address(enc_input_name, pixel_values.data_ptr())
        enc_context.set_tensor_address(enc_output_name, latents_trt.data_ptr())
        
        def run_encoder_trt():
            enc_context.execute_async_v3(stream_handle=stream.cuda_stream)
            stream.synchronize()
            
        encoder_ms = benchmark_component("TRT Encoder (Isolated)", run_encoder_trt, warmth, runs)
        
        # --- Benchmark Full TRT Pipeline ---
        # Link Encoder Output -> Decoder Input
        dec_context.set_tensor_address(dec_input_name, latents_trt.data_ptr())
        
        def run_full_trt():
            enc_context.execute_async_v3(stream_handle=stream.cuda_stream)
            dec_context.execute_async_v3(stream_handle=stream.cuda_stream)
            stream.synchronize()
            
        full_ms = benchmark_component("TRT Full Pipeline", run_full_trt, warmth, runs)
        
    return encoder_ms, decoder_ms, full_ms

def main():
    args = parse_args()
    
    # Load VAE
    if "Tiny-AutoEncoder" in args.model_path:
            logger.info("Detected Tiny AutoEncoder.")
            vae = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")

    dtype = torch.float32 if args.float32 else torch.float16
    vae.to(device="cuda", dtype=dtype)
    vae.eval()
    
    # Process Image
    logger.info(f"Processing image from {args.image_path}...")
    pixel_values = process_image(args.image_path, size=512)
    pixel_values = pixel_values.to(device="cuda", dtype=dtype)
    
    # Run Torch Benchmark
    print("-" * 50)
    torch_pipeline, torch_enc, torch_dec = benchmark_torch(vae, pixel_values)
    print("-" * 50)
    
    # Run TRT Benchmark
    trt_enc, trt_dec, trt_full = benchmark_trt(args.encoder_path, args.decoder_path, vae, pixel_values)
    
    print("\n" + "="*50)
    print("   VAE PERFORMANCE REPORT")
    print("="*50)
    
    print(f"ENCODER (Torch):      {torch_enc:6.2f} ms")
    if trt_enc:
        print(f"ENCODER (TRT):        {trt_enc:6.2f} ms  (Speedup: {torch_enc/trt_enc:.2f}x)")
    
    print("-" * 50)
    
    print(f"DECODER (Torch):      {torch_dec:6.2f} ms")
    if trt_dec:
        print(f"DECODER (TRT):        {trt_dec:6.2f} ms  (Speedup: {torch_dec/trt_dec:.2f}x)")
    
    print("-" * 50)
    
    print(f"FULL PIPELINE (Torch):  {torch_pipeline:6.2f} ms")
    if trt_full:
        print(f"FULL PIPELINE (TRT):    {trt_full:6.2f} ms  (Speedup: {torch_pipeline/trt_full:.2f}x)")
    elif trt_dec:
        # Hybrid estimated
        hybrid = torch_enc + trt_dec
        print(f"FULL PIPELINE (Hybrid): {hybrid:6.2f} ms  (Speedup: {torch_pipeline/hybrid:.2f}x)")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
