#!docker pull nvcr.io/nvidia/pytorch:20.12-py3
#!pip install -U git+https://github.com/qubvel/segmentation_models.pytorch

import torch
import os
import segmentation_models_pytorch as smp
import numpy as np


device = torch.device("cuda")
os.system('python --version')
print(torch.__version__)
print(device)


def time_model(model, dummy_input):
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    starter.record()
    _ = model(dummy_input)
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    return(curr_time)


def warm_up(model, repetitions=50):
    dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION, dtype=torch.float, device=device)
    for _ in range(repetitions):
        _ = model(dummy_input)

def get_optimal_resolution(model):
    warm_up(model)
    
    optimal_resolution = 128
    for resolution in [256, 512, 1024, 2048, 4096]:
        dummy_input = torch.randn(1, 3, resolution, resolution, dtype=torch.float, device=device)
        try:
            _ = model(dummy_input)
            optimal_resolution = resolution
        except RuntimeError as e:
            print(e)
            break
    return(optimal_resolution)

def get_latency(model, resolution):
    warm_up(model)
    
    repetitions = 300
    timings=np.zeros((repetitions,1))
    
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            dummy_input = torch.randn(1, 3, resolution, resolution, dtype=torch.float, device=device)
            timings[rep] = time_model(model, dummy_input)
        
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)    
    return mean_syn, std_syn

def get_optimal_batch_size(model):
    warm_up(model)
    
    optimal_batch_size = 1
    for batch_size in [32, 64, 128, 256, 512, 1024]:
        dummy_input = torch.randn(batch_size, 3, RESOLUTION, RESOLUTION, dtype=torch.float, device=device)
        try:
            _ = model(dummy_input)
            optimal_batch_size = batch_size
        except RuntimeError as e:
            print(e)
            break
    return(optimal_batch_size)


def get_throughput(model, batch_size, resolution):
    warm_up(model)
    
    repetitions = 100
    total_time  = 0
    with torch.no_grad():
        for rep in range(repetitions):
            dummy_input = torch.randn(batch_size, 3, resolution, resolution, dtype=torch.float, device=device)
            total_time += time_model(model, dummy_input)/1000 #to convert in second
    
    throughput =   (repetitions*batch_size)/total_time
    return(throughput)



RESOLUTION = 128

model = smp.Unet(
    encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                # model output channels (number of classes in your dataset)
)
model = model.to(device)
import torch._dynamo
torch._dynamo.reset()
model = torch.compile(model, mode = 'reduce-overhead')

for i in range(2):
    mean, std = get_latency(model, RESOLUTION)

print('Latency, average time:', mean)
print('Latency, std time:', std)

    
#optimal_batch_size = get_optimal_batch_size(model)
optimal_batch_size = 128
for i in range(2):
    throughput = get_throughput(model, optimal_batch_size, RESOLUTION)

print('Optimal Batch Size:', optimal_batch_size)
print('Final Throughput:',throughput)

#optimal_resolution = get_optimal_resolution(model)
#print('Optimal Resolution:', optimal_resolution)
#mean, std = get_latency(model, optimal_resolution)
#print('Average time:', mean)
#print((mean-std, mean+std))
