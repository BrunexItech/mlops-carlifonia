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
RUN pip install prometheus-fastapi-instrumentator

# Copy the rest of the project
COPY . .

# Create necessary directories if they do not exist
RUN mkdir -p model data/raw data/processed data/features data/splits logs

# Install AWS CLI and configure credentials
RUN apt update && apt install -y awscli wget gnupg
RUN aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
RUN aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
RUN aws configure set default.region eu-north-1

# Download model from S3
RUN aws s3 cp s3://next-mlops-california-data/dvc-store/files/md5/c9/712409b0f4ca6fac2e2995488bb362 model/linear_regression_model.pkl

# Install Prometheus
RUN wget https://github.com/prometheus/prometheus/releases/download/v2.37.0/prometheus-2.37.0.linux-amd64.tar.gz && \
    tar xvfz prometheus-*.tar.gz && \
    mv prometheus-2.37.0.linux-amd64/prometheus /usr/local/bin/ && \
    mkdir -p /etc/prometheus && \
    rm -rf prometheus-*

# Install Grafana
RUN wget -q -O - https://packages.grafana.com/gpg.key | apt-key add - && \
    echo "deb https://packages.grafana.com/oss/deb stable main" | tee -a /etc/apt/sources.list.d/grafana.list && \
    apt update && \
    apt install -y grafana

# Copy Prometheus config
COPY monitoring/prometheus.yml /etc/prometheus/prometheus.yml

# Copy Grafana dashboard
COPY monitoring/grafana-dashboard.json /etc/grafana/provisioning/dashboards/

# Expose ports (API + Prometheus + Grafana)
EXPOSE 8000 9090 3000

# Command to run all services
CMD sh -c "prometheus --config.file=/etc/prometheus/prometheus.yml & /usr/sbin/grafana-server --homepath /usr/share/grafana --config /etc/grafana/grafana.ini & uvicorn src.api.app:app --host 0.0.0.0 --port 8000"