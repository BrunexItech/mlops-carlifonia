# Dockerfile

# Base Image
FROM python:3.10-slim

# AWS credentials as build arguments
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Set Working Directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Create necessary directories if they do not exist
RUN mkdir -p model data/raw data/processed data/features data/splits

# Install AWS CLI and configure credentials
RUN apt update && apt install -y awscli
RUN aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
RUN aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
RUN aws configure set default.region eu-north-1

# Download model from S3 (UPDATED WITH CORRECT HASH)
RUN aws s3 cp s3://next-mlops-california-data/dvc-store/c9/712409b0f4ca6fac2e2995488bb362 model/linear_regression_model.pkl

# Expose the app port 
EXPOSE 8000

# Command to run Fast API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]