Unveiling the Unseen: Smart Grasping through Occlusion-Aware Semantic Segmentation

- Aperture is the distance between the two opposing fingers of the hand. So basically, how open is the hand?

<!-- python3 main.py --mode 'eval' --fcn_model 'downloads/fcn_model.pt' --reg_model 'downloads/reg_model.pt' --n_scenes 5 -->
python3 main.py --mode 'eval' --fcn_model 'save/fcn/model_9.pt' --reg_model 'downloads/reg_model.pt' --n_scenes 5

python3 main.py --mode 'fcn' --dataset_dir 'save/ppg-dataset'

python3 collect_data.py --singulation_condition --n_samples 10000 --seed 1

For the Pose-FCN Pose (Paper):
- To simplify learning of the angle θ, we account for
different pushing directions by rotating the input heightmap into
k = 16 discrete orientations (different multiplies of 22.5◦) and
then consider only horizontal pushes to the right

For the Aperture-CNN
- The depth heightmap I is first rotated according to optimal angle θ∗ and then cropped with
respect to optimal initial position p∗ to form a 64 × 64 map. The
optimal p∗ and θ∗ are produced by the Pose-FCN module.