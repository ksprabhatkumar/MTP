import torch
import json
# Import the correct class name found in your file
from models.mamba_whar import GAE_Mamba_WHAR

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize using the correct class name
    model = GAE_Mamba_WHAR(hidden_dim=64).to(device)
    model.eval()

    # B, T, N, F_dim format
    dummy_input = torch.randn(128, 125, 5, 9).to(device)

    # 1. Count Parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. Warm up GPU
    for _ in range(10):
        _ = model(dummy_input)

    # 3. Measure Latency over 50 passes
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)
    end_event.record()
    torch.cuda.synchronize()
    
    avg_latency_ms = start_event.elapsed_time(end_event) / 50.0

    # Save to JSON
    results = {
        "model_name": "GAE-Mamba-WHAR (Proposed)",
        "parameters": total_params,
        "latency_ms": avg_latency_ms
    }
    
    with open("mamba_stats.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Mamba Benchmark Saved! Params: {total_params:,} | Latency: {avg_latency_ms:.2f} ms")

if __name__ == "__main__":
    run_benchmark()