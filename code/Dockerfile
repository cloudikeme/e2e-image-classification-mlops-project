# Use a non-GPU TensorFlow base image
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt .

# Install TensorFlow and other dependencies
RUN pip install --no-cache-dir tensorflow==2.4.3 && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy only the necessary Python scripts
COPY data_ingestion.py distributed_training.py evaluate_models.py predict.py ./

# Clean up
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*