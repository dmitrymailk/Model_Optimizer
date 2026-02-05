
import argparse
import logging
import os
import time

import torch
import tensorrt as trt
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, AutoModel, UNet2DModel

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Full Pipeline (VAE + UNet): Torch vs TensorRT")
    
    # Model/Engine Paths
    parser.add_argument("--vae-path", type=str, default="fal/FLUX.2-Tiny-AutoEncoder", help="VAE Model ID")
    parser.add_argument("--vae-enc-path", type=str, default="flux_vae_tiny_trt/vae_encoder.plan", help="TRT VAE Encoder")
    parser.add_argument("--vae-dec-path", type=str, default="flux_vae_tiny_trt/vae_decoder.plan", help="TRT VAE Decoder")
    parser.add_argument("--unet-path", type=str, default="unet_trt/unet.plan", help="TRT UNet engine")
    
    parser.add_argument("--image-path", type=str, default=r"C:\programming\auto_remaster\inference_optimization\170_2x.png", help="Input image")
    parser.add_argument("--steps", type=int, default=1, help="Number of UNet steps to simulate")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 (default FP16)")
    
    return parser.parse_args()

def process_image(image_path, size=512):
    if not os.path.exists(image_path):
        # Fallback to random noise if image missing
        logger.warning(f"Image {image_path} not found, using random noise.")
        return torch.randn(1, 3, size, size)
        
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return transform(image).unsqueeze(0)

# --- Torch Benchmark ---
def get_torch_models(vae_path, device, dtype):
    # VAE
    if "Tiny-AutoEncoder" in vae_path:
        vae = AutoModel.from_pretrained(vae_path, trust_remote_code=True)
    else:
        vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
    vae.to(device=device, dtype=dtype).eval()
    
    # UNet (using same config as optimize_unet.py)
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
    unet = UNet2DModel(**unet2d_config).to(device, dtype=dtype).eval()
    unet.requires_grad_(False)
    
    return vae, unet

def benchmark_torch(vae, unet, pixel_values, steps=20, warmth=5, runs=20):
    logger.info(f"Benchmarking Torch Pipeline ({steps} steps)...")
    device = pixel_values.device
    dtype = pixel_values.dtype
    
    # Pre-compute timestep (constant for benchmark)
    timestep = torch.tensor([1.0], device=device, dtype=dtype)
    
    def run_pipeline():
        # 1. Encode
        enc_out = vae.encode(pixel_values)
        if hasattr(enc_out, "latent_dist"): latents = enc_out.latent_dist.sample()
        elif hasattr(enc_out, "latents"): latents = enc_out.latents
        else: latents = enc_out[0]
            
        # 2. Loop UNet
        for _ in range(steps):
             # Simple accumulation to prevent optimization elision, though unet(latents) replaces latents typically
             # In real diffusion, latents = scheduler.step(model_output, ...). Here we just feed output back or loop.
             # Feeding output back changes distribution, but for perf benchmark it's fine.
             # Note: Output of UNet here is (1, 128, 32, 32), same as input.
             latents = unet(latents, timestep, return_dict=False)[0]
             
        # 3. Decode
        image = vae.decode(latents, return_dict=False)[0]
        return image

    # Warmup
    for _ in range(warmth):
        with torch.no_grad(): run_pipeline()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(runs):
        with torch.no_grad(): run_pipeline()
    end.record()
    torch.cuda.synchronize()
    
    avg_ms = start.elapsed_time(end) / runs
    return avg_ms

# --- TensorRT Benchmark ---
class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
    def allocate_buffers(self, input_shapes):
        """Minimal allocation helper"""
        # input_shapes: dict of name -> shape
        pass # We'll do manual allocation in main loop for tighter control/sharing if needed

def benchmark_trt(vae_enc_path, vae_dec_path, unet_path, pixel_values, steps=20, warmth=5, runs=20):
    logger.info(f"Benchmarking TensorRT Pipeline ({steps} steps)...")
    
    if not all(os.path.exists(p) for p in [vae_enc_path, vae_dec_path, unet_path]):
        logger.error("Missing one or more TRT engines.")
        return None

    # Load Engines
    def load(path):
        logger.info(f"Loading {path}...")
        with open(path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine, engine.create_execution_context()

    enc_eng, enc_ctx = load(vae_enc_path)
    unet_eng, unet_ctx = load(unet_path)
    dec_eng, dec_ctx = load(vae_dec_path)
    
    stream = torch.cuda.Stream()
    
    # Allocations
    # 1. Image Input (1, 3, 512, 512)
    img_in_tensor = pixel_values.clone()
    
    # 2. Latents (1, 128, 32, 32) (Shared between Enc Out, UNet In/Out, Dec In)
    # We need double buffering for UNet if doing In->Out? 
    # TRT allows In-place if supported, but let's use two buffers to be safe: latents_A, latents_B.
    latents_A = torch.zeros(1, 128, 32, 32, dtype=torch.float16, device="cuda")
    latents_B = torch.zeros(1, 128, 32, 32, dtype=torch.float16, device="cuda")
    
    # 3. Timestep
    timestep_tensor = torch.tensor([1.0], dtype=torch.float16, device="cuda")
    
    # 4. Dec Output
    img_out_tensor = torch.zeros(1, 3, 512, 512, dtype=torch.float16, device="cuda")
    
    # Bindings Setup
    # Helper to set bindings by name detection
    def set_bindings(engine, context, name_map):
        # name_map: {partial_name_in_engine: tensor_ptr}
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            ptr = None
            for key, val in name_map.items():
                if key in tensor_name: # Simple substring match
                    ptr = val
                    break
            if ptr is not None:
                context.set_tensor_address(tensor_name, ptr)
                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    # Infer shape from tensor
                    # Handle scalar timestep (0D vs 1D) issue if arises
                    shape = tuple(torch.as_tensor(ptr).shape) # Hack to get shape from ptr? No, we need object.
                    # We passed tensor object in simple logic above, but here we just have address in loop
                    # Wait, we need the object to get shape.
                    pass # Done below manually
    
    # --- Encoder Context ---
    # In: image, Out: latent
    enc_ctx.set_input_shape("image", img_in_tensor.shape)
    enc_ctx.set_tensor_address("image", img_in_tensor.data_ptr())
    # Assuming encoder output is named 'latent' or similar
    # Check engine names
    enc_out_name = [enc_eng.get_tensor_name(i) for i in range(enc_eng.num_io_tensors) if enc_eng.get_tensor_mode(enc_eng.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT][0]
    enc_ctx.set_tensor_address(enc_out_name, latents_A.data_ptr())
    
    # --- UNet Context ---
    # In: sample, timestep. Out: out_sample
    unet_ctx.set_input_shape("sample", latents_A.shape)
    # Timestep
    ts_name = "timestep" # Assuming standard name
    unet_ctx.set_input_shape(ts_name, timestep_tensor.shape)
    unet_ctx.set_tensor_address(ts_name, timestep_tensor.data_ptr())
    
    # Buffers for UNet loop
    # We will swap A and B: In=A, Out=B -> In=B, Out=A
    # We need to re-bind usage in loop? execute_async_v3 allows keeping bindings if addresses don't change.
    # But if we swap buffers, addresses change.
    # Actually, efficient way:
    # Bind In=A, Out=B. Run.
    # Bind In=B, Out=A. Run.
    # Repeat.
    
    sample_name = "sample"
    out_sample_name = "out_sample"
    
    # --- Decoder Context ---
    # In: latent, Out: image
    dec_in_name = [dec_eng.get_tensor_name(i) for i in range(dec_eng.num_io_tensors) if dec_eng.get_tensor_mode(dec_eng.get_tensor_name(i)) == trt.TensorIOMode.INPUT][0]
    dec_out_name = [dec_eng.get_tensor_name(i) for i in range(dec_eng.num_io_tensors) if dec_eng.get_tensor_mode(dec_eng.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT][0]
    
    dec_ctx.set_input_shape(dec_in_name, latents_A.shape)
    dec_ctx.set_tensor_address(dec_out_name, img_out_tensor.data_ptr())

    def run_pipeline():
        # 1. Encode -> writes to latents_A
        enc_ctx.execute_async_v3(stream_handle=stream.cuda_stream)
        
        curr_in = latents_A
        curr_out = latents_B
        
        # 2. Loop
        for i in range(steps):
            unet_ctx.set_tensor_address(sample_name, curr_in.data_ptr())
            unet_ctx.set_tensor_address(out_sample_name, curr_out.data_ptr())
            unet_ctx.execute_async_v3(stream_handle=stream.cuda_stream)
            
            # Swap
            curr_in, curr_out = curr_out, curr_in
            
        # 3. Decode
        # Decoder input is whatever curr_in holds now
        dec_ctx.set_tensor_address(dec_in_name, curr_in.data_ptr())
        dec_ctx.execute_async_v3(stream_handle=stream.cuda_stream)
        
        stream.synchronize()
        return img_out_tensor

    # Warmup
    for _ in range(warmth):
        run_pipeline()
        
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(runs):
        run_pipeline()
    end.record()
    torch.cuda.synchronize()
    
    avg_ms = start.elapsed_time(end) / runs
    return avg_ms

def main():
    args = parse_args()
    
    use_fp16 = not args.fp32
    device = "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32
    
    pixel_values = process_image(args.image_path)
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    
    print(f"Benchmarking with {args.steps} UNet steps.")
    print("-" * 50)
    
    # 1. Torch
    try:
        vae, unet = get_torch_models(args.vae_path, device, dtype)
        torch_ms = benchmark_torch(vae, unet, pixel_values, steps=args.steps)
        torch_fps = 1000.0 / torch_ms if torch_ms > 0 else 0
        print(f"Torch Pipeline:   {torch_ms:8.2f} ms ({torch_fps:6.1f} FPS)")
        
        # Free memory
        del vae, unet
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Torch benchmark failed: {e}")
        torch_ms = 0
        
    print("-" * 50)
    
    # 2. TensorRT
    try:
        trt_ms = benchmark_trt(args.vae_enc_path, args.vae_dec_path, args.unet_path, pixel_values, steps=args.steps)
        if trt_ms:
            trt_fps = 1000.0 / trt_ms
            print(f"TRT Pipeline:     {trt_ms:8.2f} ms ({trt_fps:6.1f} FPS)")
    except Exception as e:
        logger.error(f"TRT benchmark failed: {e}")
        trt_ms = 0
        
    print("=" * 50)
    if torch_ms and trt_ms:
        speedup = torch_ms / trt_ms
        print(f"Overall Speedup:  {speedup:.2f}x")
    print("=" * 50)

if __name__ == "__main__":
    main()
