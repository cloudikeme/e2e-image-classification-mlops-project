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


## Running Your Image Classification Training Script (For Beginners)

This guide explains how to run the `training.py` script and verify your model is training correctly. 

**Prerequisites**

1. **Completed README Setup:** You should have already cloned the repository, created a conda environment, and installed the required packages according to the README instructions.
2. **Chosen Your Directories:** Decide where on your computer you want to save your trained model and checkpoints.  Some suggestions:

   - **Windows:**
     - `C:\Users\YourUsername\Documents\ml_projects\saved_models`
     - `C:\Users\YourUsername\Documents\ml_projects\checkpoints`

   - **macOS/Linux:**
     - `/Users/YourUsername/Documents/ml_projects/saved_models`
     - `/Users/YourUsername/Documents/ml_projects/checkpoints` 
     - Or use `~/Dev/ml_projects/saved_models` (the `~` represents your home directory)

**Steps**

1. **Create Directories (if needed):**
   - **Windows (Command Prompt):**
     ```bash
     mkdir C:\Users\YourUsername\Documents\ml_projects\saved_models
     mkdir C:\Users\YourUsername\Documents\ml_projects\checkpoints
     ```

   - **macOS/Linux (Terminal):**
     ```bash
     mkdir -p /Users/YourUsername/Documents/ml_projects/saved_models
     mkdir -p /Users/YourUsername/Documents/ml_projects/checkpoints
     ```
     - Or (shorthand): 
       ```bash
       mkdir -p ~/Dev/ml_projects/saved_models 
       mkdir -p ~/Dev/ml_projects/checkpoints
       ```

2. **Open Your Terminal/Command Prompt:**
   - **Windows:** Search for "cmd" or "Command Prompt"
   - **macOS:** Search for "Terminal"
   - **Linux:** Your terminal application will vary by distribution.

3. **Navigate to Project Directory:**
   - Use the `cd` command to navigate to the directory where your `training.py` script is located. For example:
     ```bash
     cd /Users/YourUsername/Documents/image-classification-fashion-mnist 
     ```
     (Replace with your actual directory path)

4. **Run the Training Script:** 
   - Use the following command, replacing the placeholders with your chosen directories:

     ```bash
     python training.py --saved_model_dir "C:\Users\YourUsername\Documents\ml_projects\saved_models" --checkpoint_dir "C:\Users\YourUsername\Documents\ml_projects\checkpoints" --model_type cnn
     ```
     **Important:**
      - Use forward slashes (`/`) in paths on macOS/Linux.
      - Use double quotes (`"`) around paths with spaces in them on Windows. 
      - Replace `"C:\Users\YourUsername\Documents\ml_projects\saved_models"` and `"C:\Users\YourUsername\Documents\ml_projects\checkpoints"` with your chosen model and checkpoint directories.
      - Choose the `model_type`: `cnn`, `dropout`, or `batch_norm`.

5. **Observe the Output:**
   - You should see the training progress printed to the terminal. Look for:
     - Model summary (layers, parameters)
     - Epoch numbers and training metrics (loss, accuracy) - These should generally improve over time.

6. **Verify Saved Files:**
   - After training completes, check your chosen `saved_model_dir` and `checkpoint_dir` locations. You should see files related to your saved model and checkpoints. 

**Troubleshooting:**

- **Errors:** Carefully read any error messages. Common issues include: 
  - Incorrect paths
  - Missing dependencies (revisit the README installation steps) 
- **Performance:** If training is slow, consider:
  - Using a GPU if available (requires a different TensorFlow installation).
  - Reducing the batch size in the `training.py` script.