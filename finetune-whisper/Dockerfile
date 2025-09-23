# Start from an official NVIDIA base image with CUDA and cuDNN pre-installed
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for audio processing and Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all your project files (scripts, data, models) into the container
COPY . .

# Set a default command (optional, can be overridden)
CMD ["/bin/bash"]
