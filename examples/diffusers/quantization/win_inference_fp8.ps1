$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if ($env:PATH -notlike "*$env:CUDA_HOME\bin*") {
    $env:PATH = "$env:CUDA_HOME\bin;" + $env:PATH
}

# Add TensorRT to PATH
$env:PATH = "C:\programming\auto_remaster\inference_optimization\TensorRT-10.15.1.29\bin;" + $env:PATH

# Run inference
# --onnx-load-path points to the directory where we exported the ONNX model (test_onnx)
# Check if engine exists to avoid rebuilding

# python diffusion_trt.py --model sdxl-turbo --prompt "a cinematic shot of a futuristic city at night, neon lights, 8k resolution" --onnx-load-path test_onnx\model.onnx --save-image-as output_fp8.png 

python diffusion_trt.py --model sdxl-turbo --prompt "a photo of an orange cat" --onnx-load-path test_onnx\model.onnx --save-image-as output_fp8.png --trt-engine-load-path sdxl-turbo.plan --enable-model-cpu-offload


