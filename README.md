# gemm_tflop
Simple GEMM test to calculate TFLOPs

## Clone the test script from Github repository
```
git clone https://github.com/derekngo-smc/gemm_tflop .
```

## Build docker image
Pulls and build a docker image to run GEMM test.

```
docker build -t gemm_tflop .
```

## Run docker image to calculate TFLOPs
Run docker image to test any GPU device.  In this example, GPU0 is selected to run the test.

```
docker run -it --gpus '"device=0"' gemm_tflop
   python3 -m torch.distributed.run --standalone gemm_tflop.py
```

## Run docker image to stress all GPU devices
Run docker image to test any GPU device.  In this example, GPU0 is selected to run the test.

```
docker run -it --gpus '"device=0"' gemm_tflop
   python3 -m torch.distributed.run --standalone gpu_stresstest.py
```
