# Issue about installation of pointnet2_ops_lib
## `TORCH_CUDA_ARCH_LIST`
Sometimes you may see an error message like this:
```
#11 [7/7] RUN pip install ./pointnet2_ops_lib        
#11 16.92   error: subprocess-exited-with-error      
#11 16.92                                            
#11 16.92   × python setup.py bdist_wheel did not run successfully.
#11 16.92   │ exit code: 1
#11 16.92   ╰─> [101 lines of output]
#11 16.92       No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda-12.4'
#11 16.92       running bdist_wheel

...
...
...

#11 16.92       /opt/conda/envs/HoLa-Brep/lib/python3.10/site-packages/torch/utils/cpp_extension.py:1965: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.
#11 16.92       If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
```

This error message indicates that the torch library could not find any CUDA hardware (the Docker context cannot locate CUDA hardware), resulting in the absence of any *Compute Capabilities*. Thus, you need to manually modify `setup.py` to ensure that the Docker image supports CUDA.

Please check line 19 in `setup.py`:
```
os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
```

Use the command `/usr/local/cuda-xx.x/nvcc --list-gpu-arch` to check the GPU architecture supported by your GPUs. The output may look like this:
```
compute_50
compute_52
compute_53
compute_60
compute_61
compute_62
compute_70
compute_72
compute_75
compute_80
compute_86
compute_87
compute_89
compute_90
```

According to the results of this command, you can check your GPU's architecture name **[here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)**. To specify the compute capabilities, this **[link](https://developer.nvidia.com/cuda-gpus)** will be helpful.

After specifying everything, you can edit line 19 in `setup.py`.

For instance, if your GPU is an *Nvidia 4090*.

```
os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;6.2;7.0;7.5;8.6;8.9;9.0"
```

You can check **[here](https://pytorch.org/docs/stable/cpp_extension.html)** and **[here](https://github.com/pytorch/extension-cpp/issues/71)** for more details about `TORCH_CUDA_ARCH_LIST` 