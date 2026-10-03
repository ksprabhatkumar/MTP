import subprocess
import sys
import os
import shutil

def check_gpu():
    print("=== 1. GPU Information ===")
    try:
        # Try checking for NVIDIA specifically
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            print(f"NVIDIA GPU: {result.stdout.strip()}")
            return
    except FileNotFoundError:
        pass
    
    print("nvidia-smi not found. Checking general display adapters...")
    try:
        # Fallback for all GPUs on Windows
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], capture_output=True, text=True)
        lines = [line for line in result.stdout.split('\n') if line.strip() and "Name" not in line]
        for line in lines:
            print(f"Found GPU: {line.strip()}")
    except Exception as e:
        print(f"Could not retrieve GPU info: {e}")

def check_python():
    print("\n=== 2. Current Python Version ===")
    print(f"Python: {sys.version.split()[0]}")
    print("-> MANUAL ACTION: Please tell me if you are willing/able to install Python 3.11 or 3.12 instead.")

def check_cpp_tools():
    print("\n=== 3. C++ Build Tools (MSVC) ===")
    if shutil.which("cl"):
        print("[YES] MSVC Compiler (cl.exe) is in your PATH.")
        return
        
    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio"
    ]
    
    found_vs = False
    for base_path in vs_paths:
        if os.path.exists(base_path):
            found_vs = True
            print(f"[FOUND] Visual Studio directory detected at: {base_path}")
            
    if not found_vs:
        print("[NO] Could not find Visual Studio directories.")
        print("-> You likely need to install 'Desktop development with C++' via Visual Studio Installer.")

def manual_questions():
    print("\n=== 4. Environment Preference ===")
    print("-> MANUAL ACTION: Please reply stating if you strictly want to stay on Native Windows, or if you are open to using WSL2 (Windows Subsystem for Linux).")

if __name__ == "__main__":
    check_gpu()
    check_python()
    check_cpp_tools()
    manual_questions()