"""
Weight Decay Experiment Runner for Model Inversion Defenses.

This script automates the process of testing Weight Decay (L2 Regularization)
as a defense against Knowledge-Enriched Distributional Model Inversion Attacks.
"""

import os
import json
import subprocess
import sys  # <-- 1. Add this import!

def modify_weight_decay(config_path: str, model_name: str, wd_value: float) -> None:
    # ... (Docstrings omitted for brevity, keep yours!) ...
    with open(config_path, 'r') as file:
        data = json.load(file)
        
    data[model_name]['weight_decay'] = wd_value
    
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"[INFO] Updated {model_name} weight_decay to {wd_value} in {config_path}")

def run_experiment(wd_factors: list, target_model: str = "VGG16") -> None:
    config_file = "./config/classify.json"
    
    for wd in wd_factors:
        print(f"\n{'='*50}")
        print(f"STARTING EXPERIMENT WITH WEIGHT DECAY: {wd}")
        print(f"{'='*50}\n")
        
        # 1. Update Config
        modify_weight_decay(config_file, target_model, wd)
        
        # 2. Train Target Model (Notice sys.executable here)
        print(f"--> Training Target Classifier ({target_model}) with wd={wd}...")
        subprocess.run([sys.executable, "train_classifier.py"])
        
        # 3. Train Inversion-Specific GAN (Notice sys.executable here)
        print("--> Training Inversion-Specific GAN...")
        subprocess.run([sys.executable, "k+1_gan.py"])
        
        # 4. Run Distributional Recovery Attack (Notice sys.executable here)
        print("--> Executing Distributional Recovery Attack...")
        log_file_name = f"results_wd_{wd}.txt"
        with open(log_file_name, "w") as log:
            subprocess.run([sys.executable, "recovery.py", "--model", target_model, "--improved_flag", "--dist_flag"], stdout=log)
        
        print(f"[SUCCESS] Finished experiment for weight decay = {wd}. Results saved to {log_file_name}.")

if __name__ == "__main__":
    # Baseline (1e-4) and 2 experimental groups for stronger regularization
    weight_decay_groups = [1e-4, 1e-3, 1e-2]
    
    # Run the automated pipeline
    run_experiment(weight_decay_groups, target_model="VGG16")