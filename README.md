# End-to-End Distributed MLOps Project On a KinD Kubernetes Cluster #1

## Image Classification Project

This repository contains code for training and evaluating an image classification model using TensorFlow. 

### Prerequisites

* **Operating System:** Linux, macOS, or Windows

### 1. Installing Anaconda

If you don't have Anaconda installed, download the appropriate installer for your operating system from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) and follow the instructions.

### 2. Creating a Conda Environment

1. **Open your terminal or Anaconda Prompt.**

2. **Create a new environment with Python 3.7 (compatible with TensorFlow 2.4.3):**
   ```bash
   conda create -n tf_env python=3.7
   ```

3. **Activate the new environment:**
   ```bash
   conda activate tf_env
   ```

### 3. Installing Dependencies

1. **Navigate to the project directory:**
   ```bash
   cd /path/to/your/project
   ```

2. **Install dependencies from `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```
   This will install the following packages:
   ```
   tensorflow==2.4.3
   absl-py==0.10.0
   termcolor==1.1.0
   wrapt==1.12.1
   tensorflow-datasets
   numpy==1.21
   scipy 
   matplotlib
   pandas
   ```

### 4. (Optional) GPU Acceleration

For faster training with a compatible NVIDIA GPU:

1. **Install CUDA Toolkit and cuDNN:** Download versions compatible with your TensorFlow version from the NVIDIA website.
2. **Verify GPU recognition:** After installation, run `nvidia-smi` in your terminal to check if TensorFlow recognizes your GPU. 

### 5. Running the Code

1. **Create Directories:**
   ```bash
   mkdir -p ~/Dev/mlops/saved-models
   mkdir -p ~/Dev/mlops/checkpoints
   ```

2. **Run the Training Script:**
   ```bash
   python your_script_name.py --saved_model_dir ~/Dev/mlops/saved-models --checkpoint_dir ~/Dev/mlops/checkpoints --model_type cnn
   ```
   * Replace `your_script_name.py` with the name of your Python script.
   * Adjust the paths for `--saved_model_dir` and `--checkpoint_dir` if needed. 

### Troubleshooting

* **Dependency Issues:** Create a new conda environment and reinstall packages if you encounter dependency errors.
* **GPU Not Found:** Ensure correct CUDA Toolkit and cuDNN versions are installed and compatible with your TensorFlow version.

### Additional Notes:

* This README provides a basic framework. Customize it with details about your project, dataset, model architecture, and evaluation instructions. 
* Remember to activate your conda environment (`conda activate tf_env`) before running your script or installing dependencies.
