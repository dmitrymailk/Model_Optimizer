"""
Compare PyTorch and TensorRT pipeline outputs.
Computes various metrics and generates a visual comparison.
"""
import numpy as np
from PIL import Image
import os

def load_image(path):
    """Load image and convert to numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0

def compute_metrics(img1, img2):
    """Compute comparison metrics between two images."""
    # Mean Absolute Error
    mae = np.mean(np.abs(img1 - img2))
    
    # Mean Squared Error
    mse = np.mean((img1 - img2) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse)
    else:
        psnr = float('inf')
    
    # Structural Similarity (simplified)
    mean1, mean2 = np.mean(img1), np.mean(img2)
    std1, std2 = np.std(img1), np.std(img2)
    cov = np.mean((img1 - mean1) * (img2 - mean2))
    
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2*mean1*mean2 + c1) * (2*cov + c2)) / ((mean1**2 + mean2**2 + c1) * (std1**2 + std2**2 + c2))
    
    # Per-channel stats
    channel_mae = [np.mean(np.abs(img1[:,:,c] - img2[:,:,c])) for c in range(3)]
    channel_mean_diff = [np.mean(img1[:,:,c]) - np.mean(img2[:,:,c]) for c in range(3)]
    
    return {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "channel_mae": {"R": channel_mae[0], "G": channel_mae[1], "B": channel_mae[2]},
        "channel_mean_diff": {"R": channel_mean_diff[0], "G": channel_mean_diff[1], "B": channel_mean_diff[2]},
    }

def create_comparison_image(img1, img2, output_path):
    """Create side-by-side comparison with difference map."""
    # Difference map (amplified for visibility)
    diff = np.abs(img1 - img2)
    diff_amplified = np.clip(diff * 10, 0, 1)  # Amplify 10x
    
    # Convert back to PIL
    pil1 = Image.fromarray((img1 * 255).astype(np.uint8))
    pil2 = Image.fromarray((img2 * 255).astype(np.uint8))
    pil_diff = Image.fromarray((diff_amplified * 255).astype(np.uint8))
    
    # Create combined image
    w, h = pil1.size
    combined = Image.new('RGB', (w * 3, h + 30), color=(30, 30, 30))
    
    combined.paste(pil1, (0, 30))
    combined.paste(pil2, (w, 30))
    combined.paste(pil_diff, (w * 2, 30))
    
    combined.save(output_path)
    return output_path

def main():
    torch_path = "benchmark_output_torch.png"
    trt_path = "benchmark_output_trt.png"
    ref_path = "inference_result_trt_full.png"
    
    # Check files exist
    files = {"PyTorch Benchmark": torch_path, "TRT Benchmark": trt_path, "TRT Reference": ref_path}
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"WARNING: {name} ({path}) not found!")
    
    print("Loading images...")
    img_torch = load_image(torch_path) if os.path.exists(torch_path) else None
    img_trt = load_image(trt_path) if os.path.exists(trt_path) else None
    img_ref = load_image(ref_path) if os.path.exists(ref_path) else None
    
    if img_torch is not None:
        print(f"Torch Benchmark - shape: {img_torch.shape}, range: [{img_torch.min():.3f}, {img_torch.max():.3f}]")
    if img_trt is not None:
        print(f"TRT Benchmark   - shape: {img_trt.shape}, range: [{img_trt.min():.3f}, {img_trt.max():.3f}]")
    if img_ref is not None:
        print(f"TRT Reference   - shape: {img_ref.shape}, range: [{img_ref.min():.3f}, {img_ref.max():.3f}]")
    
    print("\n" + "="*60)
    print("COMPARISON METRICS")
    print("="*60)
    
    # Compare Torch vs TRT Benchmark
    if img_torch is not None and img_trt is not None:
        print("\n--- PyTorch Benchmark vs TRT Benchmark ---")
        metrics = compute_metrics(img_torch, img_trt)
        print_metrics(metrics)
        create_comparison_image(img_torch, img_trt, "comparison_torch_vs_trt.png")
        print("Visual saved to: comparison_torch_vs_trt.png")
    
    # Compare TRT Reference vs TRT Benchmark
    if img_ref is not None and img_trt is not None:
        print("\n--- TRT Reference vs TRT Benchmark ---")
        metrics_ref = compute_metrics(img_ref, img_trt)
        print_metrics(metrics_ref)
        create_comparison_image(img_ref, img_trt, "comparison_ref_vs_trt.png")
        print("Visual saved to: comparison_ref_vs_trt.png")
    
    # Compare Torch vs TRT Reference
    if img_torch is not None and img_ref is not None:
        print("\n--- PyTorch Benchmark vs TRT Reference ---")
        metrics_torch_ref = compute_metrics(img_torch, img_ref)
        print_metrics(metrics_torch_ref)
        create_comparison_image(img_torch, img_ref, "comparison_torch_vs_ref.png")
        print("Visual saved to: comparison_torch_vs_ref.png")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if img_torch is not None and img_trt is not None:
        m = compute_metrics(img_torch, img_trt)
        assess_quality("Torch vs TRT Benchmark", m)
    if img_ref is not None and img_trt is not None:
        m = compute_metrics(img_ref, img_trt)
        assess_quality("TRT Ref vs TRT Benchmark", m)
    if img_torch is not None and img_ref is not None:
        m = compute_metrics(img_torch, img_ref)
        assess_quality("Torch vs TRT Reference", m)

def print_metrics(metrics):
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")

def assess_quality(name, metrics):
    if metrics['mae'] < 0.01:
        status = "✅ EXCELLENT"
    elif metrics['mae'] < 0.05:
        status = "✅ GOOD"
    elif metrics['mae'] < 0.1:
        status = "⚠️  FAIR"
    else:
        status = "❌ POOR"
    print(f"{name}: {status} (MAE={metrics['mae']:.4f}, PSNR={metrics['psnr']:.1f}dB)")

if __name__ == "__main__":
    main()
