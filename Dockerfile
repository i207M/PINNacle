# Bayesian Uncertainty Quantification for Meta-PINNs
# Complete reproducibility environment

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""

# Create non-root user for security
RUN useradd -m -u 1000 researcher && \
    chown -R researcher:researcher /app
USER researcher

# Default command runs quick reproduction
CMD ["python", "reproduce_all.py", "--quick"]

# Labels for metadata
LABEL maintainer="Bayesian Meta-PINN Research Team"
LABEL description="Complete reproducibility environment for Bayesian Uncertainty Quantification"
LABEL version="1.0.0"