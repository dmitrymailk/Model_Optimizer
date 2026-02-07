$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if ($env:PATH -notlike "*$env:CUDA_HOME\bin*") {
    $env:PATH = "$env:CUDA_HOME\bin;" + $env:PATH
}

# Add TensorRT to PATH
$env:PATH = "C:\programming\auto_remaster\inference_optimization\TensorRT-10.15.1.29\bin;" + $env:PATH

# Run optimization for UNet
# Model Path: checkpoint-299200
$ModelPath = "c:\programming\auto_remaster\inference_optimization\models\lbm_train_test_gap_tiny\checkpoint-299200"

# Clean output directory
if (Test-Path "unet_trt") {
    Remove-Item -Recurse -Force "unet_trt"
}

. "C:\programming\auto_remaster\venv\Scripts\Activate.ps1"; python optimize_unet.py --model-path $ModelPath --output-dir "unet_trt" --opset 18
