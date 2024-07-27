FROM python:3.7

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libopenblas-dev \
    openmpi-bin \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Set up the working directory
WORKDIR /app

# Copy the environment.yaml file into the container
COPY environment.yaml .

# Create the conda environment
RUN conda env create -f environment.yaml

# Copy only the necessary files to reduce image size 
COPY distributed_training.py ./ 

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "tf_env", "/bin/bash", "-c"]


# Verify the environment
RUN python -c "import tensorflow as tf; print(tf.__version__)" && \
    python -c "import numpy as np; print(np.__version__)"

# Set the default command to run your script
CMD ["python", "distributed_training.py", "--saved_model_dir", "/app/saved_model", "--checkpoint_dir", "/app/checkpoints", "--model_type", "cnn"]

# Set the default command to run your script
# Run script when the container launches
CMD ["python", "distributed-training.py", "--saved_model_dir", "/app/saved_model", "--checkpoint_dir", "/app/checkpoints", "--model_type", "cnn"]
CMD ["conda", "run", "--no-capture-output", "-n", "tf_env", "python", "distributed_training.py", "--saved_model_dir", "/app/saved_model", "--checkpoint_dir", "/app/checkpoints", "--model_type", "cnn"]

# Expose any necessary ports (e.g., for TensorBoard)
EXPOSE 6006