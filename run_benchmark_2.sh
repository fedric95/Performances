docker pull nvcr.io/nvidia/pytorch:23.03-py3
docker run --gpus all --ipc=host -v /home/ec2-user/SageMaker/benchmark/benchmark.py:/home/benchmark.py -it nvcr.io/nvidia/pytorch:23.03-py3 sh -c "pip install -U git+https://github.com/qubvel/segmentation_models.pytorch && python /home/benchmark.py"
