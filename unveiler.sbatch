#!/bin/bash
#SBATCH -p bullet
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mail-user=chrisantus.eze@okstate.edu
#SBATCH --mail-type=end
#SBATCH --gres=gpu:1
module load apptainer/1.2.2 cuda/12.1
export APPTAINER_BIND=$PWD
# apptainer run --nv apptainerfile.sif python3 main.py --dataset_dir 'save/pc-ou-dataset2' --mode 'unveiler' --epochs 100 --batch_size 8 --lr 1e-5 #For Unveiler 8/1e-5
# apptainer run --nv apptainerfile.sif python3 main.py --dataset_dir 'save/pc-ou-dataset' --mode 'sre' --epochs 100 --batch_size 32 --lr 1e-4 #For SRE
# apptainer run --nv apptainerfile.sif python3 main.py --dataset_dir 'save/pc-ou-dataset' --mode 'ae' --epochs 100 --batch_size 4 --lr 1e-5 #For AE
apptainer run --nv apptainerfile.sif python3 playground.py


# 16
# 1e-4

# apptainer run --nv apptainerfile.sif python3 main_act.py \
# --task_name sim_object_unveiler \
# --ckpt_dir act/ckpt \
# --policy_class ACT \
# --kl_weight 10 \
# --chunk_size 10 \
# --hidden_dim 512 \
# --batch_size 16 \
# --dim_feedforward 3200 \
# --num_epochs 4000 \
# --lr 1e-4 \
# --num_patches 10 \
# --seed 0