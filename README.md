# slowtraining
Sample code for https://github.com/tensorflow/models/issues/7395

NVidia configuration when the numbers were run:

NVIDIA-SMI 418.40.04    Driver Version: 418.40.04    CUDA Version: 10.1

GPU:0 
Name:Tesla V100-SXM2... 
Persistence-M: Off
Bus-Id: 00000000:00:1E.0 
Disp.A: Off 
Volatile GPU-Util: 
Uncorr. ECC: 0
Fan:N/A  
Temp: 46C
Perf: P0 
Pwr:Usage/Cap
Memory-Usage: 15771MiB / 16130MiB
Compute M.:Default


With slowmodel=False, GPU utilization stays around 70-80%. Times are in fast.txt

With slowmodel=True, utilization pins at 100% . Times are in slow.txt

Memory usage was the same across both.
