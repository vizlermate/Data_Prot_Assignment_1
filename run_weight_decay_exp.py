"""
Weight Decay Experiment Runner for Model Inversion Defenses.

This script automates the process of testing Weight Decay (L2 Regularization)
as a defense against Knowledge-Enriched Distributional Model Inversion Attacks.
It modifies the weight_decay parameter in the configuration file dynamically, 
trains the target models, trains the inversion-specific GANs, and evaluates 
the attack success rate.
"""

import os
import json
import subprocess

def modify_weight_decay(config_path: str, model_name: str, wd_value: float) -> None:
    """
    Modifies the JSON configuration file to update the weight decay factor.
    
    Args:
        config_path (str): The relative path to the configuration JSON file.
        model_name (str): The name of the model whose config needs updating (e.g., 'VGG16').
        wd_value (float): The new weight decay value to inject.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If the model_name is not found in the config.
    """
    with open(config_path, 'r') as file:
        data = json.load(file)
        
    data[model_name]['weight_decay'] = wd_value
    
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"[INFO] Updated {model_name} weight_decay to {wd_value} in {config_path}")

def run_experiment(wd_factors: list, target_model: str = "VGG16") -> None:
    """
    Executes the full model inversion pipeline for a list of weight decay factors.
    
    Args:
        wd_factors (list): A list of floats representing the weight decay factors to test.
        target_model (str): The model architecture to target. Defaults to "VGG16".
    """
    config_file = "./config/classify.json"
    
    for wd in wd_factors:
        print(f"\n{'='*50}")
        print(f"STARTING EXPERIMENT WITH WEIGHT DECAY: {wd}")
        print(f"{'='*50}\n")
        
        # 1. Update Config
        modify_weight_decay(config_file, target_model, wd)
        
        # 2. Train Target Model
        print(f"--> Training Target Classifier ({target_model}) with wd={wd}...")
        subprocess.run(["python", "train_classifier.py"])
        
        # 3. Train Inversion-Specific GAN
        print("--> Training Inversion-Specific GAN...")
        subprocess.run(["python", "k+1_gan.py"])
        
        # 4. Run Distributional Recovery Attack
        print("--> Executing Distributional Recovery Attack...")
        log_file_name = f"results_wd_{wd}.txt"
        with open(log_file_name, "w") as log:
            subprocess.run(["python", "recovery.py", "--model", target_model, "--improved_flag", "--dist_flag"], stdout=log)
        
        print(f"[SUCCESS] Finished experiment for weight decay = {wd}. Results saved to {log_file_name}.")

if __name__ == "__main__":
    # Baseline (1e-4) and 2 experimental groups for stronger regularization
    weight_decay_groups = [1e-4, 1e-3, 1e-2]
    
    # Run the automated pipeline
    run_experiment(weight_decay_groups, target_model="VGG16")