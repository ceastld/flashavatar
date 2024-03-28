import torch
import time

# 创建一个大张量
tensor_size = (10000, 10000)  # 10000x10000的张量
tensor_cpu = torch.randn(*tensor_size)

# 将张量移动到GPU上，并测量传输时间
start_time = time.time()
with torch.cuda.stream(torch.cuda.Stream()):
    tensor_gpu = tensor_cpu.to("cuda")
torch.cuda.synchronize()  # 等待异步数据传输完成
end_time = time.time()

# 计算传输速度
transfer_speed = tensor_cpu.numel() * tensor_cpu.element_size() / (end_time - start_time) / 1e9  # 计算传输速度（GB/s）

print("CPU到GPU传输速度:", transfer_speed, "GB/s")
