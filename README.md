Unveiling the Unseen: Smart Grasping through Occlusion-Aware Semantic Segmentation

conda create -n unveiler python=3.9.12
conda activate unveiler

- Aperture is the distance between the two opposing fingers of the hand. So basically, how open is the hand?

<!-- python3 main.py --mode 'eval' --fcn_model 'downloads/fcn_model.pt' --reg_model 'downloads/reg_model.pt' --n_scenes 5 -->
python3 main.py \
--mode 'eval' \
--reg_model 'downloads/reg_model.pt' \
--fcn_model 'save/fcn/fcn_model_17.pt' \
--n_scenes 50 \
--temporal_agg 

python3 main.py \
--dataset_dir 'save/ou-dataset' \
--mode 'fcn' \
--epochs 100 \
--batch_size 2 \
--lr 0.001

python3 collect_data.py --singulation_condition --n_samples 30000 --seed 1

python3 main_act.py \
--task_name sim_object_unveiler \
--ckpt_dir act/ckpt \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 3 \
--hidden_dim 512 \
--batch_size 1 \
--dim_feedforward 3200 \
--num_epochs 200 \
--lr 1e-5 \
--num_patches 10 \
--seed 0

python3 main_diffusion.py \
--task_name sim_object_unveiler \
--ckpt_dir diffusion/ckpt \
--enc_type resnet18 \
--chunk_size 4 \
--batch_size 8 \
--num_epochs 2000 \
--lr 1e-5 \
--seed 0

For the Pose-FCN Pose (Paper):
- To simplify learning of the angle θ, we account for
different pushing directions by rotating the input heightmap into
k = 16 discrete orientations (different multiplies of 22.5◦) and
then consider only horizontal pushes to the right

For the Aperture-CNN
- The depth heightmap I is first rotated according to optimal angle θ∗ and then cropped with
respect to optimal initial position p∗ to form a 64 × 64 map. The
optimal p∗ and θ∗ are produced by the Pose-FCN module.