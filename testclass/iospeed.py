import os
import time

def test_disk_write_speed(file_path, data):
    start_time = time.time()
    with open(file_path, 'wb') as f:
        f.write(data)
    end_time = time.time()
    return end_time - start_time

def test_disk_read_speed(file_path):
    start_time = time.time()
    with open(file_path, 'rb') as f:
        data = f.read()
    end_time = time.time()
    return end_time - start_time

# 测试写入速度
file_path = "test_file.bin"
data_to_write = b'\x00' * (1024 * 1024 * 1024)  # 写入1024MB数据
write_time = test_disk_write_speed(file_path, data_to_write)
print("写入速度（MB/s）:", 1024 / write_time)

# 测试读取速度
read_time = test_disk_read_speed(file_path)
print("读取速度（MB/s）:", 1024 / read_time)

# 删除测试文件
os.remove(file_path)
