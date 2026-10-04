import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dataset.data_loader import get_loso_dataloaders
from models.mamba_whar import GAE_Mamba_WHAR

def plot_attention_heatmap(dataset_name="realdisp", subject_id=1):
    print(f"🔍 Generating Attention Heatmap for {dataset_name.upper()} (Subject {subject_id})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup node labels
    if dataset_name == "realdisp":
        nodes = 9
        labels = ["BACK", "LUA", "RUA", "LLA", "RLA", "LT", "RT", "LC", "RC"]
    else:
        nodes = 5
        labels = ["Torso", "RA", "LA", "RL", "LL"]

    # Load Model
    model = GAE_Mamba_WHAR(num_nodes=nodes, in_features=9, hidden_dim=16, num_classes=33 if dataset_name=="realdisp" else 19)
    weights_path = f"checkpoints/mtp2_{dataset_name}_best_fold_{subject_id}.pth"
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Could not load weights from {weights_path}. Run training first!")
        return

    # Load 1 batch of Test Data
    _, _, test_loader = get_loso_dataloaders(dataset_name, subject_id, batch_size=32)
    x, _ = next(iter(test_loader))
    x = x.to(device)

    # Forward pass to grab the Dynamic Attention Matrix
    with torch.no_grad():
        _, _, A_dyn = model(x)
        
    # A_dyn shape is [B*T, N, N]. We average it over the Batch and Time dimensions
    attention_matrix = A_dyn.mean(dim=0).cpu().numpy()

    # Normalize matrix to 0-1 for plotting clarity
    attention_matrix = (attention_matrix - attention_matrix.min()) / (attention_matrix.max() - attention_matrix.min() + 1e-8)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.title(f"Learned Dynamic Sensor Correlations (A_dyn) - {dataset_name.upper()}", fontsize=14, pad=15)
    plt.xlabel("Key Sensors (Information Source)", fontsize=12)
    plt.ylabel("Query Sensors (Information Destination)", fontsize=12)
    plt.tight_layout()
    
    save_path = f"attention_heatmap_{dataset_name}_s{subject_id}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Heatmap successfully saved to {save_path}")

if __name__ == "__main__":
    plot_attention_heatmap("realdisp", 1)