import os
import torch
import main  # Imports your existing main.py directly!

def run_full_retraining_loops():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Datasets to run the ablation on (Opportunity first, then DSADS)
    datasets_to_run = ["realdisp"]
    
    # We want to run completely independent experiments for these max epochs
    target_epochs = list(range(10, 105, 10)) 

    for dataset_name in datasets_to_run:
        print(f"\n\n{'*'*70}")
        print(f"🌟 STARTING ABLATION STUDY FOR DATASET: {dataset_name.upper()}")
        print(f"{'*'*70}")
        
        for target in target_epochs:
            print(f"\n\n{'='*60}")
            print(f"🚀 {dataset_name.upper()} | FRESH EXPERIMENT: MAX EPOCHS = {target}")
            print(f"{'='*60}")
            
            # 1. Dynamically overwrite the hardcoded 'epochs' inside main.py's CONFIGS dictionary
            main.CONFIGS[dataset_name]["epochs"] = target
            
            # 2. FORCE RESTART: Delete the checkpoint so it starts from Subject 1, Epoch 1
            if os.path.exists(main.CHECKPOINT_PATH):
                os.remove(main.CHECKPOINT_PATH)
                print("[!] Cleared old checkpoints. Starting entirely from scratch.")

            # 3. Call the run_experiment function directly with NO resume state
            main.run_experiment(dataset_name, device, resume_state=None)
            
            print(f"✅ Finished full LOSO for {dataset_name.upper()} at Epoch {target} configuration.")
            print("Check results_log.txt for the final averages. Moving to next target...\n")

if __name__ == "__main__":
    run_full_retraining_loops()