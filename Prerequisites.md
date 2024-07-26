To address the issues with TensorFlow, CUDA, and TensorRT, follow these detailed steps:

1. Verify CUDA installation:
   a. Check CUDA version: `nvcc --version`
   b. Ensure CUDA is in your PATH: `echo $PATH | grep cuda`
   c. If missing, add to PATH: `export PATH=/usr/local/cuda/bin:$PATH`
   d. Add to LD_LIBRARY_PATH: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`

2. Install or update CUDA:
   a. Download CUDA Toolkit from NVIDIA website
   b. Follow installation instructions for your OS
   c. Reboot your system after installation

3. Install cuDNN:
   a. Download cuDNN from NVIDIA website (requires account)
   b. Extract and copy files to CUDA directory:
      ```
      sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
      sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
      sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
      ```

4. Install TensorRT:
   a. Download TensorRT from NVIDIA website
   b. Extract and add to PATH and LD_LIBRARY_PATH:
      ```
      export PATH=/path/to/TensorRT-7.x.x.x/bin:$PATH
      export LD_LIBRARY_PATH=/path/to/TensorRT-7.x.x.x/lib:$LD_LIBRARY_PATH
      ```

5. Reinstall TensorFlow:
   a. Uninstall current version: `pip uninstall tensorflow`
   b. Install GPU version: `pip install tensorflow-gpu`

6. Verify installation:
   a. Run Python and import TensorFlow
   b. Check GPU availability:
      ```python
      import tensorflow as tf
      print(tf.test.is_built_with_cuda())
      print(tf.test.is_gpu_available())
      ```

7. If issues persist, check compatibility:
   a. Ensure TensorFlow, CUDA, and cuDNN versions are compatible
   b. Consult TensorFlow's compatibility matrix online

8. System-wide configuration:
   a. Add CUDA paths to /etc/environment or ~/.bashrc for persistence

