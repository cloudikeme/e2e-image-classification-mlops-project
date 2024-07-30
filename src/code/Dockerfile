# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:2.4.3-gpu 

# Copy only the necessary files to reduce image size 
COPY requirements.txt ./ 

# Install dependencies before copying the rest of the project
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining project files
COPY data_ingestion.py /
COPY distributed_training.py / 
COPY evaluate_models.py /
COPY predict.py /