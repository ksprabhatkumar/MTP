import torch
import platform
import os

def get_system_info():
    print("="*40)
    print("🖥️  SYSTEM & HARDWARE ENVIRONMENT")
    print("="*40)
    print(f"OS Platform     : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version  : {platform.python_version()}")
    
    # PyTorch & CUDA Info
    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version    : {torch.version.cuda}")
        print(f"GPU Name        : {torch.cuda.get_device_name(0)}")
        print(f"GPU VRAM        : {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")
    else:
        print("GPU Name        : No NVIDIA GPU detected.")
        
    print("="*40)

if __name__ == "__main__":
    get_system_info()