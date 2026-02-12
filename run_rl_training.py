"""
Script to run RL fine-tuning of the Spatial Relationship Encoder.

Usage:
    python run_rl_training.py --epochs 500 --lr 0.00005 --seed 42

This script will:
1. Load the pre-trained SRE model
2. Initialize the PPO trainer
3. Fine-tune with RL in simulation
4. Save checkpoints to save/sre_rl/
"""

import subprocess
import sys

def main():
    cmd = [
        sys.executable,
        "main.py",
        "--mode", "sre-rl",
        "--sre_model", "save/sre/sre_model_best.pt",  # Pre-trained SRE
        "--config", "yaml/bhand.yml",  # Environment config
        "--epochs", "500",  # Number of episodes
        "--lr", "0.00005",  # Lower learning rate for fine-tuning
        "--seed", "42",
        "--num_patches", "10",
        "--patch_size", "64"
    ]
    
    print("Starting RL fine-tuning of SRE...")
    print("Command:", " ".join(cmd))
    print("-" * 80)
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
