$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if ($env:PATH -notlike "*$env:CUDA_HOME\bin*") {
    $env:PATH = "$env:CUDA_HOME\bin;" + $env:PATH
}

# Add TensorRT to PATH
$env:PATH = "C:\programming\auto_remaster\inference_optimization\TensorRT-10.15.1.29\bin;" + $env:PATH

# Run benchmark
. "C:\programming\auto_remaster\venv\Scripts\Activate.ps1"; python benchmark_vae.py
