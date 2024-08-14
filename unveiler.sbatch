#!/bin/bash
#SBATCH -p bullet
#SBATCH -t 84:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mail-user=chrisantus.eze@okstate.edu
#SBATCH --mail-type=end
#SBATCH --gres=gpu:1
module load apptainer/1.2.2 cuda/12.1
export APPTAINER_BIND=$PWD
# apptainer run --nv apptainerfile.sif python3 main.py --dataset_dir 'save/pc-ou-dataset2' --mode 'fcn' --epochs 100 --batch_size 8 --lr 1e-5
# apptainer run --nv apptainerfile.sif python3 main.py --dataset_dir 'save/ppg-ou-dataset' --mode 'fcn' --epochs 100 --batch_size 4 --lr 0.001
# apptainer run --nv apptainerfile.sif python3 modify_dataset.py
# apptainer run --nv apptainerfile.sif python3 playground.py
# apptainer run --nv apptainerfile.sif python3 collect_data.py --singulation_condition --n_samples 30000 --seed 1321

apptainer run --nv apptainerfile.sif python3 main_act.py \
--task_name sim_object_unveiler \
--ckpt_dir act/ckpt \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 8 \
--hidden_dim 512 \
--batch_size 4 \
--dim_feedforward 3200 \
--num_epochs 200 \
--lr 1e-5 \
--num_patches 10 \
--seed 0
