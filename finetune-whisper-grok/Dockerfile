FROM finetune_whisper_lora:v2  # Or your base

# Copy cuDNN libs from build context (host's version)
COPY cuda_libs/lib64/ /usr/local/cuda/lib64/

# Update library cache and set env (override old cuDNN vars)
RUN ldconfig /usr/local/cuda/lib64 && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> /etc/environment

# Update cuDNN env vars to match latest (optional, for consistency)
ENV NV_CUDNN_VERSION=9.14.0  # Adjust to your host's version
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9  # If upgraded to 9.x
ENV NV_CUDNN_PACKAGE=libcudnn9=9.14.0-1+cuda12.0

# Reinstall/upgrade pip packages if needed (e.g., for newer cuDNN Python bindings)
RUN pip install --upgrade nvidia-cudnn-cu12

# Your existing installs/scripts...
WORKDIR /app
CMD ["/bin/bash"]
