FROM ubuntu:22.04

# Update the package list and install Python 3 and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install numpy and torch using pip
RUN pip3 install numpy torch

# Set the working directory
WORKDIR /workdir

# Copy the Python script to the working directory
COPY gemm_tflop.py .

# Execute the Python script
CMD ["python3", "-m", "torch.distributed.run", "--standalone", "./gemm_tflop.py"]
