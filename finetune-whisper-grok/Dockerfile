FROM finetune_whisper_lora:v2

# Set pip config for custom index (add early, before any pip installs)
RUN pip config set global.index-url https://dso-nexus.*****.vn/repository/itd_pypi/simple && \
    pip config set global.trusted-host dso-nexus.*****.vn

# Copy cuDNN libs from build context (host's version at /usr/local/cuda/lib64)
COPY cuda_libs/lib64/ /usr/local/cuda/lib64/

# Update library cache and append to LD_LIBRARY_PATH
RUN ldconfig /usr/local/cuda/lib64 && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> /etc/environment

# Update cuDNN env vars to match latest (adjust to your host's version, e.g., 9.x)
ENV NV_CUDNN_VERSION=9.14.0
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9
ENV NV_CUDNN_PACKAGE=libcudnn9=9.14.0-1+cuda12.0

# Install only missing libraries for notebook workflow (evaluate for metrics, jiwer for WER, gradio for demo)
RUN pip install evaluate jiwer gradio

# Set working directory
WORKDIR /app

# Use NVIDIA entrypoint for GPU support
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
