# system_evaluation.py
import psutil
import os
import time

class SystemEvaluator:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.cpu_times_start = None

    def start_monitoring(self):
        """Start timing and CPU usage tracking."""
        self.start_time = time.time()
        self.cpu_times_start = self.process.cpu_times()

    def log_resources(self, prefix):
        """Log CPU percentage, cumulative CPU time, and memory usage."""
        # CPU percentage over a 1-second interval for sustained usage
        cpu_percent = self.process.cpu_percent(interval=1.0)
        # Cumulative CPU time (user + system) since process start
        cpu_times = self.process.cpu_times()
        cpu_time_total = cpu_times.user + cpu_times.system
        memory_mb = self.process.memory_info().rss / 1024 / 1024 
        print(f"{prefix} - CPU: {cpu_percent:.2f}%, CPU Time: {cpu_time_total:.2f}s, Memory: {memory_mb:.2f} MB")

    def end_monitoring(self, label):
        """End monitoring, log resources, and return duration and CPU delta."""
        if self.start_time is None or self.cpu_times_start is None:
            raise ValueError("Monitoring not started.")
        
        duration = time.time() - self.start_time
        cpu_times_end = self.process.cpu_times()
        cpu_time_delta = (cpu_times_end.user + cpu_times_end.system) - \
                         (self.cpu_times_start.user + self.cpu_times_start.system)
        
        cpu_percent = self.process.cpu_percent(interval=1.0)
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        print(f"{label} - Duration: {duration:.2f}s, CPU Time Delta: {cpu_time_delta:.2f}s, "
              f"CPU: {cpu_percent:.2f}%, Memory: {memory_mb:.2f} MB")
        
        self.start_time = None
        self.cpu_times_start = None
        return duration, cpu_time_delta