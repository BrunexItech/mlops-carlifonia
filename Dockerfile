# Dockerfile

#Base Image
FROM python:3.10-slim

#set Working Directory
WORKDIR /app

#Copy requirements first for better catching
COPY requirements.txt .

#Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#Copy the rest of the project
COPY . .

#Create necessary directories if they do not exist
RUN mkdir -p model data/raw data/processed data/features data/splits

#Expose the app port 
EXPOSE 8000

# Add these lines BEFORE the CMD line
RUN apt update && apt install -y awscli
RUN aws configure set default.region eu-north-1
RUN aws s3 cp s3://next-mlops-california-data/dvc-store/[MD5_HASH] model/linear_regression_model.pkl

#Command to run Fast api
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
