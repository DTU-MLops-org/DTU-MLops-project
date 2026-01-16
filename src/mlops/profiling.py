import torch
from mlops.model import Model
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

model = Model()
inputs = torch.randn(1, 3, 224, 224)

with profile(
    activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/model")
) as prof:
    for i in range(10):
        model(inputs)
        prof.step()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
