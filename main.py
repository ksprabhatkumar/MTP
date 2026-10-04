import os
# --- WINDOWS TRITON BUG FIX ---
if os.environ.get("CUDA_PATH") is None:
    os.environ["CUDA_PATH"] = "C:\\" 
# ------------------------------

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from dataset.data_loader import get_loso_dataloaders, get_dataset_config
from models.mamba_whar import GAE_Mamba_WHAR
from utils.metrics import calculate_metrics
from utils.logger import log_results, log_summary

CONFIGS = {
    "dsads": {
        "epochs": 100, "batch_size": 64, 
        "lr": 5e-4, "hidden_size": 16, "weight_decay": 5e-4,    
        "dropout": 0.3, "patience": 20, "lambda_mse": 0.01       
    },
    "opportunity": {
        "epochs": 100, "batch_size": 256, 
        "lr": 1e-3, "hidden_size": 8, "weight_decay": 5e-4, 
        "dropout": 0.4, "patience": 20, "lambda_mse": 0.05       
    },
    "realdisp": {
        "epochs": 50, "batch_size": 64, 
        "lr": 1e-4, "hidden_size": 16, "weight_decay": 5e-4, 
        "dropout": 0.5, "patience": 20, "lambda_mse": 0.1  
    }
}

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0.5, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = label_smoothing 
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.smoothing > 0.0:
            num_classes = inputs.size(1)
            smooth_targets = torch.full_like(inputs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            log_pt = F.log_softmax(inputs, dim=1)
            ce_loss = -(smooth_targets * log_pt).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

class EarlyStopping:
    def __init__(self, patience=15, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_f1):
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_f1
            self.counter = 0

def temporal_cutout(x, max_cutout_ratio=0.1, prob=0.1):
    if torch.rand(1).item() > prob:
        return x
    B, T, N, F = x.shape
    cutout_len = int(T * max_cutout_ratio)
    start_idx = torch.randint(0, T - cutout_len, (1,)).item()
    mask = torch.ones_like(x, device=x.device)
    mask[:, start_idx : start_idx + cutout_len, :, :] = 0.0
    return x * mask

def drop_random_sensors(x, max_drops=1, drop_prob=0.1):
    B, T, N, F_dim = x.shape
    mask = torch.ones((B, 1, N, 1), device=x.device)
    for b in range(B):
        if torch.rand(1).item() < drop_prob:
            num_drops = torch.randint(1, max_drops + 1, (1,)).item()
            drop_indices = torch.randperm(N)[:num_drops]
            mask[b, 0, drop_indices, 0] = 0.0
    return x * mask

def train_one_epoch(model, dataloader, criterion_cls, optimizer, device, lambda_mse, args):
    model.train()
    total_loss, total_cls, total_mse = 0, 0, 0
    
    for x_clean, y in tqdm(dataloader, desc="Training (n-2 pool)", leave=False):
        x_clean, y = x_clean.to(device), y.to(device)
        optimizer.zero_grad()
        
        x_aug = temporal_cutout(x_clean, max_cutout_ratio=0.1, prob=0.2)
        x_corrupt = drop_random_sensors(x_aug, max_drops=1, drop_prob=0.2)
        
        # 🌟 Pass ablation arguments
        logits, x_recon, _ = model(x_corrupt, ablate_phys=args.ablate_phys, ablate_learn=args.ablate_learn, ablate_dyn=args.ablate_dyn)
        
        loss_cls = criterion_cls(logits, y)                
        loss_mse = F.mse_loss(x_recon, x_clean)            
        loss = loss_cls + (lambda_mse * loss_mse)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_mse += loss_mse.item()
        
    return total_loss / len(dataloader), total_cls / len(dataloader), total_mse / len(dataloader)

def evaluate(model, dataloader, criterion_cls, device, lambda_mse=2.0, drop_test=0, args=None):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x_clean = x.clone()
            
            if drop_test > 0:
                B, T, N, F_dim = x.shape
                mask = torch.ones((B, 1, N, 1), device=device)
                for b in range(B):
                    drop_indices = torch.randperm(N)[:drop_test]
                    mask[b, 0, drop_indices, 0] = 0.0
                x = x * mask

            logits, x_recon, _ = model(x, ablate_phys=args.ablate_phys, ablate_learn=args.ablate_learn, ablate_dyn=args.ablate_dyn)
            
            loss_cls = criterion_cls(logits, y)
            
            if drop_test == 0:
                loss_mse = F.mse_loss(x_recon, x_clean)
                loss = loss_cls + (lambda_mse * loss_mse)
            else:
                loss = loss_cls
                
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    val_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    val_acc, val_f1 = calculate_metrics(all_labels, all_preds) if len(all_preds) > 0 else (0,0)
    return val_acc, val_f1, val_loss

def run_experiment(dataset_name, device, args):
    hyperparams = CONFIGS[dataset_name]
    config = get_dataset_config(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"🚀 MTP-2: DYNAMIC GRAPH ARCHITECTURE ({dataset_name.upper()})")
    if args.ablate_phys or args.ablate_learn or args.ablate_dyn:
        print(f"⚠️ ABLATION MODE: Phys={not args.ablate_phys}, Learn={not args.ablate_learn}, Dyn={not args.ablate_dyn}")
    print(f"{'='*60}")
    
    start_subject = 1
    loso_acc, loso_f1 = [], []
    learning_curve_history = {} 
    
    curve_file = f"n2_split_logs_{dataset_name}.json"

    for subject_id in range(start_subject, config['total_subjects'] + 1):
        print(f"\n--- FOLD {subject_id}/{config['total_subjects']} ---")
        
        train_loader, internal_val_loader, test_loader = get_loso_dataloaders(
            dataset_name, subject_id, batch_size=hyperparams["batch_size"], test_on_mutual=False
        )
        
        if dataset_name.lower() == "realdisp":
            _, _, test_loader_mutual = get_loso_dataloaders(
                dataset_name, subject_id, batch_size=hyperparams["batch_size"], test_on_mutual=True
            )
        else:
            test_loader_mutual = None
        
        model = GAE_Mamba_WHAR(
            num_nodes=config['num_nodes'], 
            in_features=config['in_features'], 
            hidden_dim=hyperparams["hidden_size"],
            num_classes=config['num_classes'],
            dropout=hyperparams["dropout"]
        ).to(device)
        
        criterion_cls = FocalLoss(gamma=0.5, label_smoothing=0.05) 
        optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=hyperparams["lr"], steps_per_epoch=1, 
            epochs=hyperparams["epochs"], pct_start=0.3, div_factor=1000.0, final_div_factor=100.0
        )
        
        early_stopping = EarlyStopping(patience=hyperparams["patience"])
        best_acc, best_f1 = 0.0, 0.0
        
        subject_key = f"Subject_{subject_id}"
        if subject_key not in learning_curve_history:
            learning_curve_history[subject_key] = {
                "train_acc": [], "val_acc": [], "val_f1": [],
                "train_loss": [], "val_loss": [], "focal_loss": [], "mse_loss": []
            }
            
        print("[*] Calculating Epoch 0 (Untrained Baseline)...")
        train_acc_0, _, train_loss_0 = evaluate(model, internal_val_loader, criterion_cls, device, hyperparams["lambda_mse"], drop_test=0, args=args)
        val_acc_0, val_f1_0, val_loss_0 = evaluate(model, test_loader, criterion_cls, device, hyperparams["lambda_mse"], drop_test=0, args=args)
        
        learning_curve_history[subject_key]["train_acc"].append(train_acc_0 * 100)
        learning_curve_history[subject_key]["val_acc"].append(val_acc_0 * 100)
        learning_curve_history[subject_key]["val_f1"].append(val_f1_0 * 100)
        learning_curve_history[subject_key]["train_loss"].append(train_loss_0)
        learning_curve_history[subject_key]["val_loss"].append(val_loss_0)
        learning_curve_history[subject_key]["focal_loss"].append(train_loss_0) 
        learning_curve_history[subject_key]["mse_loss"].append(0.0)
        
        print(f"Ep 000 | LR: 0.000000 | Tr Loss: {train_loss_0:.3f} | Val Loss: {val_loss_0:.3f} | Tr Acc: {train_acc_0*100:.2f}% | Val Acc: {val_acc_0*100:.2f}%")
        
        for epoch in range(1, hyperparams["epochs"] + 1):
            train_loss, t_cls, t_mse = train_one_epoch(
                model, train_loader, criterion_cls, optimizer, device, hyperparams["lambda_mse"], args
            )
            
            train_acc, _, _ = evaluate(
                model, internal_val_loader, criterion_cls, device, hyperparams["lambda_mse"], drop_test=0, args=args
            )
            
            val_acc, val_f1, val_loss = evaluate(
                model, test_loader, criterion_cls, device, hyperparams["lambda_mse"], drop_test=0, args=args
            )
            
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            learning_curve_history[subject_key]["train_acc"].append(train_acc * 100)
            learning_curve_history[subject_key]["val_acc"].append(val_acc * 100)
            learning_curve_history[subject_key]["val_f1"].append(val_f1 * 100)
            learning_curve_history[subject_key]["train_loss"].append(train_loss)
            learning_curve_history[subject_key]["val_loss"].append(val_loss)
            learning_curve_history[subject_key]["focal_loss"].append(t_cls)
            learning_curve_history[subject_key]["mse_loss"].append(t_mse)
            
            with open(curve_file, "w") as f:
                json.dump(learning_curve_history, f, indent=4)
            
            if val_f1 > best_f1:
                best_acc, best_f1 = val_acc, val_f1
                torch.save(model.state_dict(), f"checkpoints/mtp2_{dataset_name}_best_fold_{subject_id}.pth")
                
            print(f"Ep {epoch:03d} | LR: {current_lr:.6f} | Tr Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Tr Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
            
            early_stopping(val_f1)
            if early_stopping.early_stop:
                print(f"🛑 Early stopping triggered at epoch {epoch}!")
                break
            
        print(f"✅ Best IDEAL for Subject {subject_id}: Acc = {best_acc*100:.2f}%, F1 = {best_f1*100:.2f}%")
        
        print(f"\n🧪 Testing Sensor Robustness on Subject {subject_id}...")
        model.load_state_dict(torch.load(f"checkpoints/mtp2_{dataset_name}_best_fold_{subject_id}.pth"))
        
        _, f1_clean, _ = evaluate(model, test_loader, criterion_cls, device, hyperparams["lambda_mse"], drop_test=0, args=args)
        _, f1_1drop, _ = evaluate(model, test_loader, criterion_cls, device, hyperparams["lambda_mse"], drop_test=1, args=args)
        _, f1_2drop, _ = evaluate(model, test_loader, criterion_cls, device, hyperparams["lambda_mse"], drop_test=2, args=args)
        
        if test_loader_mutual is not None:
            acc_mut, f1_mutual, _ = evaluate(model, test_loader_mutual, criterion_cls, device, hyperparams["lambda_mse"], drop_test=0, args=args)
            print(f"🔄 MUTUAL DISPLACEMENT F1: {f1_mutual*100:.2f}%")
        
        robustness_data = (f1_clean*100, f1_1drop*100, f1_2drop*100)
        log_results(dataset_name, subject_id, best_acc * 100, best_f1 * 100, hyperparams, robustness=robustness_data)
        
        loso_acc.append(best_acc)
        loso_f1.append(best_f1)

    if len(loso_acc) > 0:
        avg_acc = np.mean(loso_acc) * 100
        avg_f1 = np.mean(loso_f1) * 100
        log_summary(dataset_name, avg_acc, avg_f1)
        print(f"\n🎉 FINISHED {dataset_name.upper()} | Avg Acc: {avg_acc:.2f}% | Avg F1: {avg_f1:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTP-2 Dynamic Graph Experiment Runner")
    parser.add_argument("--dataset", type=str, required=True, help="Choose dataset: dsads, opportunity, realdisp, or all")
    
    # 🌟 NEW ABLATION ARGUMENTS
    parser.add_argument("--ablate_phys", action="store_true", help="Turn off the Physical Skeleton Graph")
    parser.add_argument("--ablate_learn", action="store_true", help="Turn off the Global Learnable Graph")
    parser.add_argument("--ablate_dyn", action="store_true", help="Turn off the Dynamic Attention Graph")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on DEVICE: {device}")
    os.makedirs("checkpoints", exist_ok=True)
    
    if args.dataset.lower() == "all":
        datasets_to_run = ["dsads", "opportunity", "realdisp"]
    else:
        datasets_to_run = [args.dataset.lower()]

    for d_name in datasets_to_run:
        run_experiment(d_name, device, args)