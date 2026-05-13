from pathlib import Path
import os
import time
import gc
import threading
import torch
import psutil
import pandas as pd

RESULTS_DIR = Path("experiments_results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_PATH = RESULTS_DIR / "results_ram_analysis.csv" # "results_vgg16_ram_analysis.csv"
SUMMARY_RESULTS_PATH = RESULTS_DIR / "summary_ram_analysis.csv"

process = psutil.Process(os.getpid())


def get_ram_mb():
    return process.memory_info().rss / (1024 ** 2)


def get_gpu_memory_mb():
    if not torch.cuda.is_available():
        return {
            "gpu_allocated_mb": None,
            "gpu_reserved_mb": None,
            "gpu_peak_allocated_mb": None,
            "gpu_peak_reserved_mb": None,
        }

    return {
        "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def measure_memory_usage(stage_name, operation, model_type=None, sample_interval=0.05):
    cleanup_memory()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ram_before = get_ram_mb()
    peak_ram = ram_before
    stop_sampling = False

    def sampler():
        nonlocal peak_ram, stop_sampling
        while not stop_sampling:
            current_ram = get_ram_mb()
            if current_ram > peak_ram:
                peak_ram = current_ram
            time.sleep(sample_interval)

    thread = threading.Thread(target=sampler)
    thread.daemon = True

    start_time = time.time()
    thread.start()

    result = None
    error = None

    try:
        result = operation()
    except Exception as exc:
        error = exc
    finally:
        stop_sampling = True
        thread.join()

    elapsed_time = time.time() - start_time
    ram_after = get_ram_mb()
    gpu = get_gpu_memory_mb()

    metrics = {
        "model_type": model_type,
        "stage": stage_name,
        "ram_before_mb": ram_before,
        "ram_after_mb": ram_after,
        "ram_peak_mb": peak_ram,
        "ram_peak_increase_mb": peak_ram - ram_before,
        "time_seconds": elapsed_time,
        **gpu,
    }

    if error is not None:
        raise error

    return result, metrics


def print_metrics(metrics):
    if metrics.get("model_type"):
        print(f"Model: {metrics['model_type']}")

    print(f"Etap: {metrics['stage']}")
    print(f"RAM przed: {metrics['ram_before_mb']:.2f} MB")
    print(f"RAM po: {metrics['ram_after_mb']:.2f} MB")
    print(f"RAM peak: {metrics['ram_peak_mb']:.2f} MB")
    print(f"Wzrost peak RAM: {metrics['ram_peak_increase_mb']:.2f} MB")
    print(f"Czas: {metrics['time_seconds']:.2f} s")

    if metrics["gpu_peak_allocated_mb"] is not None:
        print(f"GPU peak allocated: {metrics['gpu_peak_allocated_mb']:.2f} MB")
        print(f"GPU peak reserved: {metrics['gpu_peak_reserved_mb']:.2f} MB")

def export_to_csv(df: pd.DataFrame, filename: str):
    output_path = RESULTS_DIR / filename
    df.to_csv(output_path, index=False)