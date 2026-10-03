import os
import torch
from torch.utils.data import Dataset, DataLoader

class HARDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataset_config(dataset_name):
    configs = {
        "dsads": {"num_nodes": 5, "in_features": 9, "num_classes": 19, "total_subjects": 8},
        "opportunity": {"num_nodes": 5, "in_features": 9, "num_classes": 18, "total_subjects": 4},
        "realdisp": {"num_nodes": 9, "in_features": 9, "num_classes": 33, "total_subjects": 17}
    }
    return configs[dataset_name.lower()]

def get_loso_dataloaders(dataset_name, test_subject_id, batch_size=128, test_on_mutual=False):
    base_path = f"dataset/processed/{dataset_name.lower()}"
    
    if not os.path.exists(f"{base_path}_x.pt"):
        raise FileNotFoundError(f"Processed tensors not found for {dataset_name}!")

    # 🌟 Added weights_only=True to silence warnings
    x = torch.load(f"{base_path}_x.pt", weights_only=True).float() 
    y = torch.load(f"{base_path}_y.pt", weights_only=True).long()
    p = torch.load(f"{base_path}_p.pt", weights_only=True).long()

    config = get_dataset_config(dataset_name)
    total_subjects = config["total_subjects"]

    test_mask = (p == test_subject_id)
    internal_val_id = test_subject_id + 1 if test_subject_id < total_subjects else 1
    internal_val_mask = (p == internal_val_id)
    train_mask = ~(test_mask | internal_val_mask)

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[internal_val_mask], y[internal_val_mask]
    
    if test_on_mutual and dataset_name.lower() == "realdisp":
        mut_path = "dataset/processed/realdisp_mutual"
        if not os.path.exists(f"{mut_path}_x.pt"):
            return None, None, None
            
        # 🌟 Added weights_only=True here as well
        x_mut = torch.load(f"{mut_path}_x.pt", weights_only=True).float()
        y_mut = torch.load(f"{mut_path}_y.pt", weights_only=True).long()
        p_mut = torch.load(f"{mut_path}_p.pt", weights_only=True).long()
        
        test_mask_mut = (p_mut == test_subject_id)
        x_test, y_test = x_mut[test_mask_mut], y_mut[test_mask_mut]
    else:
        x_test, y_test = x[test_mask], y[test_mask]

    mean = x_train.mean(dim=(0, 1, 2), keepdim=True)
    std = x_train.std(dim=(0, 1, 2), keepdim=True) + 1e-8 
    
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    if len(x_test) > 0:
        x_test = (x_test - mean) / std

    train_loader = DataLoader(HARDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    internal_val_loader = DataLoader(HARDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(HARDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    return train_loader, internal_val_loader, test_loader