# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:2.4.3-gpu 

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files to reduce image size 
COPY requirements.txt ./ 

# Install dependencies before copying the rest of the project
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining project files
COPY distributed_training.py ./ 
COPY evaluate_models.py ./
COPY predict.py ./


# Make port 8888 available (adjust if needed)
EXPOSE 8888

# Define environment variable (optional, customize as needed)
ENV NAME=World

# Run script when the container launches
CMD ["python", "distributed_training.py", "--saved_model_dir", "/app/saved_model", "--checkpoint_dir", "/app/checkpoints", "--model_type", "cnn"]