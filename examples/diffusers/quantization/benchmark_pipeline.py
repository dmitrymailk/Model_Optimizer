import os
import torch
import numpy as np
import tensorrt as trt
import itertools
import time
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    UNet2DModel,
    AutoencoderTiny,
)
from flux2_tiny_autoencoder import Flux2TinyAutoEncoder

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda"
WEIGHT_DTYPE = torch.float16
RESOLUTION = 512
NUM_STEPS = 1
IMAGE_INDEX = 170
SCALING_FACTOR = 0.13025 

# Torch Paths
CHECKPOINT_PATH = r"c:\programming\auto_remaster\inference_optimization\models\lbm_train_test_gap_tiny\checkpoint-299200"

# TRT Paths
TRT_ENCODER_PATH = "flux_vae_tiny_trt/vae_encoder.plan"
TRT_DECODER_PATH = "flux_vae_tiny_trt/vae_decoder.plan"
TRT_UNET_PATH = "unet_trt/unet.plan"

# Benchmark Config
WARMUP_ROUNDS = 5
BENCHMARK_ROUNDS = 200

# -----------------------------------------------------------------------------
# TensorRT Engine Wrappers
# -----------------------------------------------------------------------------
class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
    def _load_engine(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"TRT Engine not found at {path}")
        with open(path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())
            
    def infer(self, feed_dict):
        # Create NEW context for each call to prevent buffer corruption
        context = self.engine.create_execution_context()
        
        bindings = [None] * self.engine.num_io_tensors
        outputs = {}
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(tensor_name)
            
            if mode == trt.TensorIOMode.INPUT:
                if tensor_name not in feed_dict:
                     raise ValueError(f"Missing input '{tensor_name}'")
                
                tensor = feed_dict[tensor_name]
                if tensor.dtype != torch.float16:
                    tensor = tensor.to(dtype=torch.float16)
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                    
                context.set_input_shape(tensor_name, tensor.shape)
                context.set_tensor_address(tensor_name, tensor.data_ptr())
                bindings[i] = tensor.data_ptr()
            else:
                shape = context.get_tensor_shape(tensor_name)
                # Resolve dynamic shapes
                resolved_shape = list(shape)
                if -1 in resolved_shape:
                     batch_size = list(feed_dict.values())[0].shape[0]
                     if resolved_shape[0] == -1: resolved_shape[0] = batch_size
                
                output_tensor = torch.empty(tuple(resolved_shape), dtype=torch.float16, device="cuda")
                context.set_tensor_address(tensor_name, output_tensor.data_ptr())
                bindings[i] = output_tensor.data_ptr()
                outputs[tensor_name] = output_tensor

        context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs

# -----------------------------------------------------------------------------
# Benchmark Logic
# -----------------------------------------------------------------------------
def run_torch_pipeline(vae, unet, input_tensor, num_steps):
    # Create fresh scheduler
    scheduler = FlowMatchEulerDiscreteScheduler()
    
    with torch.no_grad():
        # Encode
        encoded_source = vae.encode(input_tensor, return_dict=False)
        z_source = encoded_source * vae.config.scaling_factor
        
        sample = z_source
        
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        scheduler.set_timesteps(sigmas=sigmas, device=DEVICE)
        
        # Denoise
        for i, t in enumerate(scheduler.timesteps):
            if hasattr(scheduler, "scale_model_input"):
                 denoiser_input = scheduler.scale_model_input(sample, t)
            else:
                 denoiser_input = sample
                 
            unet_input = torch.cat([denoiser_input, z_source], dim=1)
            t_batch = t.to(DEVICE).repeat(unet_input.shape[0])
            pred = unet(unet_input, t_batch, return_dict=False)[0]
            sample = scheduler.step(pred, t, sample, return_dict=False)[0]
        
        # Decode
        decoded = vae.decode(sample / vae.config.scaling_factor, return_dict=False)
        output = decoded.clamp(-1, 1)
        return output

def run_trt_pipeline(enc_engine, dec_engine, unet_engine, input_tensor, num_steps):
    # Create fresh scheduler
    scheduler = FlowMatchEulerDiscreteScheduler()
    
    # Encode
    enc_out = enc_engine.infer({"image": input_tensor})
    z_source_raw = list(enc_out.values())[0]
    z_source = z_source_raw * SCALING_FACTOR
    
    sample = z_source
    
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=DEVICE)
    
    # Denoise
    for i, t in enumerate(scheduler.timesteps):
        if hasattr(scheduler, "scale_model_input"):
             denoiser_input = scheduler.scale_model_input(sample, t)
        else:
             denoiser_input = sample
             
        cat_input = torch.cat([denoiser_input, z_source], dim=1)
        t_tensor = t.to(DEVICE).view(1).half()
        
        # UNet Inference
        unet_out = unet_engine.infer({"sample": cat_input, "timestep": t_tensor})
        pred = list(unet_out.values())[0]
        
        sample = scheduler.step(pred, t, sample, return_dict=False)[0]
        
    # Decode
    latents_to_decode = sample / SCALING_FACTOR
    dec_out = dec_engine.infer({"latent": latents_to_decode})
    output_image = list(dec_out.values())[0]
    output = output_image.clamp(-1, 1)
    return output

def benchmark(name, func, *args):
    print(f"\nBenchmarking {name}...")
    
    # Warmup
    print(f"  Warmup ({WARMUP_ROUNDS} rounds)...")
    for _ in range(WARMUP_ROUNDS):
        func(*args)
    torch.cuda.synchronize()
    
    # Measure
    print(f"  Running {BENCHMARK_ROUNDS} rounds...")
    latencies = []
    last_output = None
    
    for _ in range(BENCHMARK_ROUNDS):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        last_output = func(*args)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # ms
        
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    print(f"  {name}: Avg Latency = {avg_latency:.2f} ms +/- {std_latency:.2f} ms")
    return avg_latency, last_output

def save_image(tensor, path):
    # tensor: (1, 3, H, W) or (3, H, W) range [-1, 1]
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Denormalize
    image = tensor.cpu().float() * 0.5 + 0.5
    image = image.clamp(0, 1)
    
    # To PIL
    pil_image = transforms.ToPILImage()(image)
    pil_image.save(path)
    print(f"Saved result to {path}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    
    # 1. Prepare Data
    print("Loading data...")
    dataset_name = "dim/render_nfs_4screens_5_sdxl_1_wan_mix"
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    item = next(itertools.islice(dataset, IMAGE_INDEX, None))
    orig_source_pil = item["input_image"].convert("RGB")
    
    train_transforms = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_tensor = train_transforms(orig_source_pil).unsqueeze(0).to(DEVICE).half()
    
    # 2. Init Torch Models
    print("Loading Torch models...")
    vae_torch = Flux2TinyAutoEncoder.from_pretrained(
        "fal/FLUX.2-Tiny-AutoEncoder",
        torch_dtype=WEIGHT_DTYPE,
    ).to(DEVICE)
    vae_torch.eval()
    
    unet_torch = UNet2DModel.from_pretrained(
        CHECKPOINT_PATH,
        subfolder="unet",
        torch_dtype=WEIGHT_DTYPE,
        use_safetensors=True
    ).to(DEVICE)
    unet_torch.eval()
    
    scheduler = FlowMatchEulerDiscreteScheduler()
    
    # 3. Init TRT Engines
    print("Loading TensorRT engines...")
    enc_engine = TRTEngine(TRT_ENCODER_PATH)
    dec_engine = TRTEngine(TRT_DECODER_PATH)
    unet_engine = TRTEngine(TRT_UNET_PATH)
    
    # 4. Run Benchmarks
    torch_latency, torch_output = benchmark("PyTorch Pipeline", run_torch_pipeline, 
                              vae_torch, unet_torch, input_tensor, NUM_STEPS)
    save_image(torch_output, "benchmark_output_torch.png")
    
    trt_latency, trt_output = benchmark("TensorRT Pipeline", run_trt_pipeline, 
                            enc_engine, dec_engine, unet_engine, input_tensor, NUM_STEPS)
    save_image(trt_output, "benchmark_output_trt.png")
    
    # 5. Report
    print("\n---------------------------------------------------------")
    print("Final Results (End-to-End Latency)")
    print("---------------------------------------------------------")
    print(f"PyTorch:  {torch_latency:.2f} ms")
    print(f"TensorRT: {trt_latency:.2f} ms")
    print(f"Speedup:  {torch_latency / trt_latency:.2f}x")
    print("---------------------------------------------------------")
