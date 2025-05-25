FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Set CUDA runtime library path so bitsandbytes can find libcudart.so
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

# Install Python and system packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-setuptools \
    libopenmpi-dev git-all && \
    ln -s /usr/bin/python3 /usr/bin/python

# Install Python dependencies
RUN pip install --no-cache-dir pandas numpy scikit-learn matplotlib seaborn jupyterlab \
torch torchvision transformers datasets accelerate trl wandb peft pillow bitsandbytes

# Define environment variable
# ENV HF_TOKEN=

# Keep the container running and provide an interactive shell
CMD ["bash"]
