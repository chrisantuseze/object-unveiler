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
# apptainer run --nv apptainerfile.sif python3 main.py --dataset_dir 'save/pc-ou-dataset2' --mode 'fcn' --epochs 100 --batch_size 4 --lr 0.001
# apptainer run --nv apptainerfile.sif python3 main.py --dataset_dir 'save/ppg-ou-dataset' --mode 'fcn' --epochs 100 --batch_size 4 --lr 0.001
apptainer run --nv apptainerfile.sif python3 generate_scene_mask.py
