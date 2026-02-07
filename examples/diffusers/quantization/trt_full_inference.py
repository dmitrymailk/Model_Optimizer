
import os
import torch
import numpy as np
import tensorrt as trt
import itertools
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from diffusers import FlowMatchEulerDiscreteScheduler

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda"
WEIGHT_DTYPE = torch.float16 # TRT engines expected to handle FP16
RESOLUTION = 512
NUM_STEPS = 1
IMAGE_INDEX = 170
SCALING_FACTOR = 0.13025 

# TRT Paths
TRT_ENCODER_PATH = "flux_vae_tiny_trt/vae_encoder.plan"
TRT_DECODER_PATH = "flux_vae_tiny_trt/vae_decoder.plan"
TRT_UNET_PATH = "unet_trt/unet.plan"

OUTPUT_IMAGE_PATH = "inference_result_trt_full.png"

# -----------------------------------------------------------------------------
# TensorRT Wrapper
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
        # feed_dict maps input_name -> torch_tensor
        bindings = [None] * self.engine.num_io_tensors
        outputs = {}
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(tensor_name)
            
            if mode == trt.TensorIOMode.INPUT:
                if tensor_name not in feed_dict:
                     print(f"Warning: Missing input '{tensor_name}', expected: {feed_dict.keys()}")
                     raise ValueError(f"Missing input '{tensor_name}'")
                
                tensor = feed_dict[tensor_name]
                
                # Enforce FP16
                if tensor.dtype != torch.float16:
                    tensor = tensor.to(dtype=torch.float16)
                
                # Ensure contiguous
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                    
                self.context.set_input_shape(tensor_name, tensor.shape)
                self.context.set_tensor_address(tensor_name, tensor.data_ptr())
                bindings[i] = tensor.data_ptr()
            else:
                # Output
                shape = self.context.get_tensor_shape(tensor_name)
                # Output - Force FP16
                torch_dtype = torch.float16
                
                # Resolving dynamic shapes if needed
                resolved_shape = list(shape)
                if -1 in resolved_shape:
                     # Heuristic: use batch size from first input
                     batch_size = list(feed_dict.values())[0].shape[0]
                     if resolved_shape[0] == -1: resolved_shape[0] = batch_size
                
                output_tensor = torch.empty(tuple(resolved_shape), dtype=torch_dtype, device="cuda")
                self.context.set_tensor_address(tensor_name, output_tensor.data_ptr())
                bindings[i] = output_tensor.data_ptr()
                outputs[tensor_name] = output_tensor

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # 1. Load Dataset
    print(f"Loading dataset item {IMAGE_INDEX}...")
    dataset_name = "dim/render_nfs_4screens_5_sdxl_1_wan_mix"
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    item = next(itertools.islice(dataset, IMAGE_INDEX, None))
    orig_source_pil = item["input_image"].convert("RGB")
    
    # 2. Load Engines
    print("Loading TensorRT Engines...")
    try:
        enc_engine = TRTEngine(TRT_ENCODER_PATH)
        dec_engine = TRTEngine(TRT_DECODER_PATH)
        unet_engine = TRTEngine(TRT_UNET_PATH)
    except Exception as e:
        print(f"Error loading engines: {e}")
        exit(1)
        
    print("Engines loaded.")
    
    # Scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    
    # 3. Preprocessing
    print("Preprocessing...")
    train_transforms = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    c_t = train_transforms(orig_source_pil).unsqueeze(0).to(DEVICE).half() # FP16 input
    
    # 4. Inference Pipeline
    print(f"Running Full TRT Pipeline ({NUM_STEPS} steps)...")
    
    sigmas = np.linspace(1.0, 1 / NUM_STEPS, NUM_STEPS)
    noise_scheduler.set_timesteps(sigmas=sigmas, device=DEVICE)
    
    # A. Encode
    print("  Encoding (TRT)...")
    enc_out = enc_engine.infer({"image": c_t})
    z_source_raw = list(enc_out.values())[0] # "latent"
    
    z_source = z_source_raw * SCALING_FACTOR
    
    # B. Diffusion Loop
    sample = z_source
    
    for i, t in enumerate(noise_scheduler.timesteps):
        if hasattr(noise_scheduler, "scale_model_input"):
             denoiser_input = noise_scheduler.scale_model_input(sample, t)
        else:
             denoiser_input = sample
             
        # Concatenate (128 + 128 = 256 channels)
        cat_input = torch.cat([denoiser_input, z_source], dim=1)
        
        # Timestep tensor
        # UNet TRT expects timestep as tensor (1,)
        # Check export script: dummy_timestep was scalar tensor
        t_tensor = t.to(DEVICE).view(1) 
        if t_tensor.dtype != torch.float16:
            t_tensor = t_tensor.half() # Match engine precision if needed
            
        # Infer UNet
        # Input names from export: "sample", "timestep"
        unet_out = unet_engine.infer({"sample": cat_input, "timestep": t_tensor})
        pred = list(unet_out.values())[0] # "out_sample"
        
        # Scheduler Step
        sample = noise_scheduler.step(pred, t, sample, return_dict=False)[0]
        
    # C. Decode
    print("  Decoding (TRT)...")
    latents_to_decode = sample / SCALING_FACTOR
    dec_out = dec_engine.infer({"latent": latents_to_decode})
    output_image = list(dec_out.values())[0] # "image"
    
    # Post-process
    output_image = output_image.clamp(-1, 1)
    
    # Check shape for PIL
    if output_image.dim() == 4:
        image_tensor = output_image[0]
    else:
        image_tensor = output_image
        
    print(f"  Decoder Output Shape: {output_image.shape}")
    
    output_pil = transforms.ToPILImage()(image_tensor.cpu().float() * 0.5 + 0.5)
    
    print(f"Saving to {OUTPUT_IMAGE_PATH}")
    output_pil.save(OUTPUT_IMAGE_PATH)
    print("Success.")
