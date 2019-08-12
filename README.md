# slowtraining
Sample code for https://github.com/tensorflow/models/issues/7395

With slowmodel=False, GPU utilization stays around 70-80%. Times are in fast.txt

With slowmodel=True, utilization pins at 100% . Times are in slow.txt

Memory usage was the same across both.
