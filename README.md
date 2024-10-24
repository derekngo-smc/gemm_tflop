# gemm_tflop
Simple GEMM test to calculate TFLOPs

# Clone the test script from Github repository
git clone https://github.com/derekngo-smc/gemm_tflop .

# Build docker image
docker build -t gemm_tflop .

# Run docker image to calculate TFLOPs
docker run -it --gpus all --net host --cap-add SYS_ADMIN gemm_tflop
