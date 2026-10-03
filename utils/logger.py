import os
from datetime import datetime

def log_results(dataset_name, subject_id, acc, f1, hyperparams, robustness=None, filepath="results_log.txt"):
    """Appends the results of an experiment to a text file with hyperparameters and robustness metrics."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (f"[{timestamp}] Dataset: {dataset_name.upper()} | Subj: {subject_id} | Acc: {acc:.2f}% | F1: {f1:.2f}% | "
                 f"Params: Ep={hyperparams['epochs']}, BS={hyperparams['batch_size']}, LR={hyperparams['lr']}\n")
    
    # 🌟 NEW: Log the MTP-1 Robustness Benchmark if provided!
    if robustness:
        log_entry += (f"   --> ROBUSTNESS BENCHMARK | 0 Dead: {robustness[0]:.2f}% F1 | "
                      f"1 Dead: {robustness[1]:.2f}% F1 | 2 Dead: {robustness[2]:.2f}% F1\n")
    
    with open(filepath, "a") as f:
        f.write(log_entry)
        
def log_summary(dataset_name, avg_acc, avg_f1, filepath="results_log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] >>> FINAL AVG FOR {dataset_name.upper()} | Acc: {avg_acc:.2f}% | F1: {avg_f1:.2f}%\n\n"
    with open(filepath, "a") as f:
        f.write(log_entry)