
import os
import torch
import numpy as np
import diffusers
import itertools
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
NUM_STEPS = 1 # User set this to 1
IMAGE_INDEX = 170

CHECKPOINT_PATH = r"c:\programming\auto_remaster\inference_optimization\models\lbm_train_test_gap_tiny\checkpoint-299200"
OUTPUT_IMAGE_PATH = "inference_result.png"

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # 1. Load Dataset
    print(f"Loading dataset item {IMAGE_INDEX}...")
    dataset_name = "dim/render_nfs_4screens_5_sdxl_1_wan_mix"
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    # Get the 170th image
    item = next(itertools.islice(dataset, IMAGE_INDEX, None))
    orig_source_pil = item["input_image"].convert("RGB")
    
    # 2. Load Models
    print("Loading models...")
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    
    vae = Flux2TinyAutoEncoder.from_pretrained(
        "fal/FLUX.2-Tiny-AutoEncoder",
        torch_dtype=WEIGHT_DTYPE,
    ).to(DEVICE)
    vae.requires_grad_(False)
    vae.eval()
    
    print(f"Loading UNet from {CHECKPOINT_PATH}...")
    unet = UNet2DModel.from_pretrained(
        CHECKPOINT_PATH,
        subfolder="unet",
        torch_dtype=WEIGHT_DTYPE,
        use_safetensors=True
    ).to(DEVICE)
    unet.eval()
    
    # 3. Preprocessing
    print("Preprocessing...")
    train_transforms = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    c_t = train_transforms(orig_source_pil).unsqueeze(0).to(vae.dtype).to(DEVICE)
    
    # 4. Inference
    print(f"Running inference with {NUM_STEPS} steps...")
    
    sigmas = np.linspace(1.0, 1 / NUM_STEPS, NUM_STEPS)
    noise_scheduler.set_timesteps(sigmas=sigmas, device=DEVICE)
    
    with torch.no_grad():
        encoded_source = vae.encode(c_t, return_dict=False)
        z_source = encoded_source * vae.config.scaling_factor
        
        sample = z_source
        
        for i, t in enumerate(noise_scheduler.timesteps):
            if hasattr(noise_scheduler, "scale_model_input"):
                 denoiser_input = noise_scheduler.scale_model_input(sample, t)
            else:
                 denoiser_input = sample
                 
            unet_input = torch.cat([denoiser_input, z_source], dim=1)
            t_batch = t.to(DEVICE).repeat(unet_input.shape[0])
            pred = unet(unet_input, t_batch, return_dict=False)[0]
            sample = noise_scheduler.step(pred, t, sample, return_dict=False)[0]
        
        print("Decoding...")
        decoded = vae.decode(sample / vae.config.scaling_factor, return_dict=False)
        output_image = decoded.clamp(-1, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu().float() * 0.5 + 0.5)
        
    print(f"Saving result to {OUTPUT_IMAGE_PATH}")
    output_pil.save(OUTPUT_IMAGE_PATH)
    print("Done.")
