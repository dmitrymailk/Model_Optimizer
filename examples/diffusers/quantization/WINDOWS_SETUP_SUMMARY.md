# Windows Quantization Setup Summary

To run `win_quantize_model.ps1` successfully with GPU support on Windows, the following key steps were taken:

## 1. Prerequisites
- **Visual Studio Build Tools 2022**: Install "Desktop development with C++" workload.
- **CUDA Toolkit 12.8**: Must match the PyTorch CUDA version (strictly 12.x like 12.8, not 13.x).

## 2. Environment Configuration
The `win_quantize_model.ps1` script was modified to explicitly set `CUDA_HOME` before running:
```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if ($env:PATH -notlike "*$env:CUDA_HOME\bin*") {
    $env:PATH = "$env:CUDA_HOME\bin;" + $env:PATH
}
```

## 3. Manual Patch for Compilation Error
To fix the `error C2872: 'std': ambiguous symbol` during extension compilation, we patched the PyTorch header file:
**File:** `venv\Lib\site-packages\torch\include\torch\csrc\dynamo\compiled_autograd.h`

**Change (approx line 1110):**
```cpp
// OLD
#if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))

// NEW (Patched)
#if defined(_WIN32)
```
This forces the compiler to correctly identify the platform as Windows and skip incompatible code.

4. **Экономия памяти (для 8GB GPU)**:
   Использовать флаг `--cpu-offloading` (квантование) или `--enable-model-cpu-offload` (инференс).

## 5. Inference (FP8)
To run inference with the quantized model:
1. Ensure `TensorRT` is in your PATH.
2. Use `win_inference_fp8.ps1`.
   
   Мы добавили `--enable-model-cpu-offload` в скрипт, чтобы избежать ошибки `OutOfMemory` при загрузке движка TensorRT.

```powershell
python diffusion_trt.py --model sdxl-turbo ... --trt-engine-load-path sdxl-turbo.plan --enable-model-cpu-offload
```
