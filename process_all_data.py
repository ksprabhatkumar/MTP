import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import Counter

# --- OPPORTUNITY CONSTANTS ---
OPP_LABELS = {
    0: 0, 406516: 1, 406517: 2, 404516: 3, 404517: 4, 406520: 5, 404520: 6,
    406505: 7, 404505: 8, 406519: 9, 404519: 10, 406511: 11, 404511: 12,
    406508: 13, 404508: 14, 408512: 15, 407521: 16, 405506: 17
}
FEATURE_COLS_OPP = list(range(37, 46)) + list(range(50, 59)) + list(range(63, 72)) + list(range(76, 85)) + list(range(89, 98))
LABEL_COL_OPP = 249

# --- REALDISP CONSTANTS ---
NODE_COLS_REALDISP = (
    list(range(28, 37)) +   # 0: BACK
    list(range(41, 50)) +   # 1: LUA
    list(range(15, 24)) +   # 2: RUA
    list(range(54, 63)) +   # 3: LLA
    list(range(2, 11)) +    # 4: RLA
    list(range(93, 102)) +  # 5: LT
    list(range(80, 89)) +   # 6: RT
    list(range(106, 115)) + # 7: LC
    list(range(67, 76))     # 8: RC
)
LABEL_COL_REALDISP = 119

def save_tensors(name, data, labels, persons):
    os.makedirs("dataset/processed", exist_ok=True)
    X_tensor = torch.tensor(np.array(data), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(labels), dtype=torch.long)
    P_tensor = torch.tensor(np.array(persons), dtype=torch.long)

    torch.save(X_tensor, f"dataset/processed/{name}_x.pt")
    torch.save(Y_tensor, f"dataset/processed/{name}_y.pt")
    torch.save(P_tensor, f"dataset/processed/{name}_p.pt")
    print(f"✅ Saved {name.upper()}: X {X_tensor.shape}, Y {Y_tensor.shape}, P {P_tensor.shape}\n")

def process_dsads():
    print("Processing DSADS dataset (OPTIMIZED - Z-Score Normalization)...")
    base_path = os.path.join("har_data", "DSADS", "data")
    if not os.path.exists(base_path): return

    all_data, all_labels, all_persons = [], [], []
    for activity_id in tqdm(range(1, 20), desc="Processing DSADS"):
        for person_id in range(1, 9):
            for segment_id in range(1, 61):
                file_path = os.path.join(base_path, f"a{activity_id:02d}", f"p{person_id}", f"s{segment_id:02d}.txt")
                try:
                    data_matrix = np.loadtxt(file_path, delimiter=',')
                    all_data.append(data_matrix.reshape(125, 5, 9))
                    all_labels.append(activity_id - 1)
                    all_persons.append(person_id)
                except Exception:
                    pass

    if len(all_data) == 0: return
    X_np = np.array(all_data, dtype=np.float32)
    
    means = np.mean(X_np, axis=(0, 1), keepdims=True)
    stds = np.std(X_np, axis=(0, 1), keepdims=True)
    stds[stds == 0] = 1.0 
    X_normalized = (X_np - means) / stds
    print(f"Data Normalized! Global Mean: {np.mean(X_normalized):.4f}, Std: {np.std(X_normalized):.4f}")
    
    save_tensors("dsads", X_normalized, all_labels, all_persons)

def process_opportunity():
    print("Processing OPPORTUNITY dataset (OPTIMIZED - Z-Score Normalization)...")
    base_path = os.path.join("har_data", "OPPORTUNITY", "dataset")
    if not os.path.exists(base_path): return

    all_windows, all_labels, all_persons = [], [], []
    window_size, step_size = 24, 12

    for subject_id in range(1, 5):
        files = [f for f in os.listdir(base_path) if f.startswith(f"S{subject_id}-") and f.endswith(".dat")]
        for file_name in tqdm(files, desc=f"Subject {subject_id}"):
            file_path = os.path.join(base_path, file_name)
            
            df = pd.read_csv(file_path, sep=r'\s+', header=None)
            
            features = df[FEATURE_COLS_OPP].copy()
            labels = df[LABEL_COL_OPP].copy()
            
            features = features.interpolate(method='linear', limit_direction='both').fillna(0).values
            labels = labels.values
            mapped_labels = np.array([OPP_LABELS.get(lbl, 0) for lbl in labels])
            
            for start in range(0, len(features) - window_size, step_size):
                window_data = features[start : start + window_size]
                window_labels = mapped_labels[start : start + window_size]
                most_common_label = Counter(window_labels).most_common(1)[0][0]
                
                all_windows.append(window_data.reshape(window_size, 5, 9))
                all_labels.append(most_common_label)
                all_persons.append(subject_id)

    X_np = np.array(all_windows, dtype=np.float32)
    
    means = np.mean(X_np, axis=(0, 1), keepdims=True)
    stds = np.std(X_np, axis=(0, 1), keepdims=True)
    stds[stds == 0] = 1.0 
    X_normalized = (X_np - means) / stds
    print(f"Data Normalized! Global Mean: {np.mean(X_normalized):.4f}, Std: {np.std(X_normalized):.4f}")
    
    save_tensors("opportunity", X_normalized, all_labels, all_persons)

def process_realdisp():
    print("Processing REALDISP dataset (IDEAL & MUTUAL DISPLACEMENT)...")
    base_path = os.path.join("har_data", "REALDISP")
    if not os.path.exists(base_path): return

    modes = ["ideal", "mutual"]
    
    for mode in modes:
        all_windows, all_labels, all_persons = [], [], []
        window_size, step_size = 50, 25 
        
        found_any = False
        for subject_id in range(1, 18):
            file_paths_to_process = []
            
            if mode == "ideal":
                fpath = os.path.join(base_path, f"subject{subject_id}_ideal.log")
                if os.path.exists(fpath):
                    file_paths_to_process.append(fpath)
            else:
                # Based on directory contents, mutual files are mutual4, mutual5, mutual6, mutual7
                for mut_idx in [4, 5, 6, 7]:
                    fpath = os.path.join(base_path, f"subject{subject_id}_mutual{mut_idx}.log")
                    if os.path.exists(fpath):
                        file_paths_to_process.append(fpath)
                        
            if not file_paths_to_process:
                if mode == "mutual":
                    print(f"   ⚠️ Could not find MUTUAL data for Subject {subject_id}")
                continue
                
            found_any = True
            
            for file_path in file_paths_to_process:
                try:
                    df = pd.read_csv(file_path, sep='\t|,|\s+', header=None, engine='python')
                except Exception:
                    continue
                    
                features = df[NODE_COLS_REALDISP].copy()
                labels = df[LABEL_COL_REALDISP].copy()
                
                features = features.interpolate(method='linear', limit_direction='both').fillna(0).values
                labels = labels.values
                
                for start in range(0, len(features) - window_size, step_size):
                    window_labels = labels[start : start + window_size]
                    most_common_label = Counter(window_labels).most_common(1)[0][0]
                    
                    if most_common_label == 0:
                        continue
                        
                    window_data = features[start : start + window_size]
                    window_data_reshaped = window_data.reshape(window_size, 9, 9)
                    
                    all_windows.append(window_data_reshaped)
                    all_labels.append(int(most_common_label) - 1)
                    all_persons.append(subject_id)

        if not found_any:
            print(f"❌ Skipping {mode.upper()} - No valid files found in {base_path}!\n")
            continue

        if len(all_windows) > 0:
            X_np = np.array(all_windows, dtype=np.float32)
            
            means = np.mean(X_np, axis=(0, 1), keepdims=True)
            stds = np.std(X_np, axis=(0, 1), keepdims=True)
            stds[stds == 0] = 1.0 
            X_normalized = (X_np - means) / stds
            
            save_name = "realdisp" if mode == "ideal" else "realdisp_mutual"
            print(f"Data Normalized ({mode.upper()})! Global Mean: {np.mean(X_normalized):.4f}, Std: {np.std(X_normalized):.4f}")
            save_tensors(save_name, X_normalized, all_labels, all_persons)

if __name__ == "__main__":
    print("🚀 Starting Data Processing (OPTIMIZED MODE)...\n")
    process_dsads()
    process_opportunity()
    process_realdisp()
    print("🎉 All datasets processed & Normalized!")