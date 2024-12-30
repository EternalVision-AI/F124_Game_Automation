import psutil
import time



while True:
    cpu_percent = process.cpu_percent(interval=1)
    memory_info = process.memory_info()
    
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    
