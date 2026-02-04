$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if ($env:PATH -notlike "*$env:CUDA_HOME\bin*") {
    $env:PATH = "$env:CUDA_HOME\bin;" + $env:PATH
}
# не работает на windows
# python quantize.py --model sdxl-turbo --format int8 --batch-size 1 --calib-size 2 --alpha 0.8 --n-steps 20 --model-dtype BFloat16 --trt-high-precision-dtype BFloat16 --quantized-torch-ckpt-save-path ./test.pt --onnx-dir test_onnx

python quantize.py --model sdxl-turbo --model-dtype BFloat16 --trt-high-precision-dtype BFloat16 --format fp8 --batch-size 1 --calib-size 1 --quantize-mha --n-steps 1 --quantized-torch-ckpt-save-path ./test.pt --collect-method default --onnx-dir test_onnx --cpu-offloading