
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import onnx
import tensorrt as trt
from diffusers import FluxPipeline
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize Flux VAE Decoder")
    parser.add_argument("--model-path", type=str, default="black-forest-labs/FLUX.1-dev", help="HuggingFace model ID or local path")
    parser.add_argument("--output-dir", type=str, default="flux_vae_trt", help="Output directory for ONNX and TRT engine")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--fp16", action="store_true", help="Export in FP16 (requires GPU)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for optimization profile")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension (H=W) for export. Default 64 (512px image / 8). Use 32 for Tiny AutoEncoder on 512px.")
    return parser.parse_args()

def export_onnx(vae, output_path, opset=17, fp16=False, channels=16, latent_dim=64):
    logger.info(f"Exporting VAE Decoder to ONNX at {output_path}...")
    
    # Flux VAE config: 16 channels in. 
    # User requested strictly 512x512 image -> 512/8 = 64x64 latents.
    # Batch size 1.
    B = 1
    C = channels
    H = latent_dim
    W = latent_dim
    
    device = "cuda" if fp16 else "cpu"
    dtype = torch.float16 if fp16 else torch.float32
    vae.to(device=device, dtype=dtype)
    vae.eval()
    
    dummy_input = torch.randn(B, C, H, W, device=device, dtype=dtype)
    
    # Static shapes - no dynamic axes needed for specific resolution optimization
    dynamic_axes = None
    
    # Wrapper to only call decoder
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
            
        def forward(self, x):
            # VAE decode expects latent, return_dict=False returns tuple
            return self.vae.decode(x, return_dict=False)[0]

    model_wrapper = VAEDecoderWrapper(vae)

    torch.onnx.export(
        model_wrapper,
        (dummy_input,),
        output_path,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True
    )
    logger.info("ONNX export successful.")

def export_encoder_onnx(vae, output_path, opset=17, fp16=False, image_size=512):
    logger.info(f"Exporting VAE Encoder to ONNX at {output_path}...")
    
    B = 1
    C = 3
    H = image_size
    W = image_size
    
    device = "cuda" if fp16 else "cpu"
    dtype = torch.float16 if fp16 else torch.float32
    vae.to(device=device, dtype=dtype)
    vae.eval()
    
    dummy_input = torch.randn(B, C, H, W, device=device, dtype=dtype)
    
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
            
        def forward(self, x):
            # Encode
            encoded_output = self.vae.encode(x)
            
            # Extract latents
            if hasattr(encoded_output, "latent_dist"):
                # For ONNX export, we can't easily sample random noise unless we pass it in.
                # For benchmarking/inference stability, we often use the mode (mean).
                latents = encoded_output.latent_dist.mode() 
            elif hasattr(encoded_output, "latents"):
                 latents = encoded_output.latents
            else:
                 latents = encoded_output[0]
            
            # Scale
            if hasattr(self.vae.config, "scaling_factor") and self.vae.config.scaling_factor is not None:
                 latents = latents * self.vae.config.scaling_factor
                 
            return latents

    model_wrapper = VAEEncoderWrapper(vae)

    torch.onnx.export(
        model_wrapper,
        (dummy_input,),
        output_path,
        input_names=["image"],
        output_names=["latent"],
        dynamic_axes=None, # Static shapes for now
        opset_version=opset,
        do_constant_folding=True
    )
    logger.info("Encoder ONNX export successful.")

def build_trt_engine(onnx_path, engine_path, fp16=False, verbose=False):
    logger.info(f"Building TensorRT engine at {engine_path}...")
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    if fp16:
        if not builder.platform_has_fast_fp16:
            logger.warning("Platform does not support fast FP16, falling back to FP32.")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
    
    # Parse ONNX
    # Use parse_from_file to correctly handle external data files
    if not parser.parse_from_file(onnx_path):
        for error in range(parser.num_errors):
            logger.error(parser.get_error(error))
        raise RuntimeError("Failed to parse ONNX file")
            
    # Optimization inputs
    profile = builder.create_optimization_profile()
    
    input_tensor = network.get_input(0)
    input_name = input_tensor.name if input_tensor else "input"
    
    if input_tensor:
        in_dims = input_tensor.shape
        logger.info(f"Parsed ONNX input dims for '{input_name}': {in_dims}")
        
        c_val = in_dims[1] if in_dims[1] > 0 else 16
        h_val = in_dims[2] if in_dims[2] > 0 else 64
        w_val = in_dims[3] if in_dims[3] > 0 else 64
        
        # B=1 for this specific task
        profile.set_shape(input_name, (1, c_val, h_val, w_val), (1, c_val, h_val, w_val), (1, c_val, h_val, w_val))
    else:
        # Fallback
        logger.warning(f"Could not determine input dims, using fallback for '{input_name}'")
        profile.set_shape(input_name, (1, 16, 64, 64), (1, 16, 64, 64), (1, 16, 64, 64))
        
    config.add_optimization_profile(profile)
    
    # Build
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Engine build failed")
            
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        logger.info("TensorRT engine build successful.")
        
    except Exception as e:
        logger.error(f"Error building engine: {e}")
        raise

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths for Decoder
    dec_onnx_path = os.path.join(args.output_dir, "vae_decoder.onnx")
    dec_engine_path = os.path.join(args.output_dir, "vae_decoder.plan")
    
    # Paths for Encoder
    enc_onnx_path = os.path.join(args.output_dir, "vae_encoder.onnx")
    enc_engine_path = os.path.join(args.output_dir, "vae_encoder.plan")
    
    logger.info(f"Loading Flux Pipeline from {args.model_path}...")
    try:
        from diffusers import AutoencoderKL, AutoModel
        
        # Check for Tiny AutoEncoder
        if "Tiny-AutoEncoder" in args.model_path:
            logger.info("Detected Tiny AutoEncoder. Loading with trust_remote_code=True...")
            vae = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
        else:
            subfolder = "vae"
            try:
                vae = AutoencoderKL.from_pretrained(args.model_path, subfolder=subfolder)
            except:
                logger.info("Could not load with subfolder='vae', trying direct load...")
                vae = AutoencoderKL.from_pretrained(args.model_path)
            
    except Exception as e:
        logger.error(f"Failed to load VAE: {e}")
        sys.exit(1)
        
    # Debug: Inspect VAE
    logger.info(f"VAE Config: {vae.config}")
    
    # Try to determine correct channels from config
    channels = 16 # Default Flux
    if hasattr(vae.config, "latent_channels"):
        channels = vae.config.latent_channels
    elif hasattr(vae.config, "in_channels"):
        channels = vae.config.in_channels
        
    logger.info(f"Using {channels} channels for export.")
    
    # --- ENCODER ---
    logger.info("--- Processing Encoder ---")
    export_encoder_onnx(vae, enc_onnx_path, opset=args.opset, fp16=args.fp16, image_size=512)
    if os.path.exists(enc_onnx_path):
        build_trt_engine(enc_onnx_path, enc_engine_path, fp16=args.fp16)
    else:
        logger.error("Encoder ONNX export failed.")

    # --- DECODER ---
    logger.info("--- Processing Decoder ---")
    export_onnx(vae, dec_onnx_path, opset=args.opset, fp16=args.fp16, channels=channels, latent_dim=args.latent_dim)
    if os.path.exists(dec_onnx_path):
        build_trt_engine(dec_onnx_path, dec_engine_path, fp16=args.fp16)
    else:
        logger.error("Decoder ONNX export failed.")
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
