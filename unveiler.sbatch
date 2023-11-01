#!/bin/bash
#SBATCH -p bullet
#SBATCH -t 120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mail-user=chrisantus.eze@okstate.edu
#SBATCH --mail-type=end
#SBATCH --gres=gpu:1
module load apptainer/1.2.2 cuda/12.1
export APPTAINER_BIND=$PWD
# apptainer run --nv apptainerfile.sif python3 main.py --dataset_dir 'save/ppg-ou-dataset' --mode 'fcn' --epochs 100 --batch_size 4 --lr 0.0001 #python3 main.py --mode 'fcn' --dataset_dir 'save/ppg-dataset-5k' --epochs 100 --batch_size 5
apptainer run --nv apptainerfile.sif python3 main.py --dataset_dir 'save/ppg-dataset-5k' --mode 'fcn' --epochs 100 --batch_size 1 --lr 0.0001 #python3 main.py --mode 'fcn' --dataset_dir 'save/ppg-dataset-5k' --epochs 100 --batch_size 5