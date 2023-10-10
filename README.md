Unveiling the Unseen: Smart Grasping through Occlusion-Aware Semantic Segmentation

- Aperture is the distance between the two opposing fingers of the hand. So basically, how open is the hand?

<!-- python3 main.py --mode 'eval' --fcn_model 'downloads/fcn_model.pt' --reg_model 'downloads/reg_model.pt' --n_scenes 5 -->
python3 main.py --mode 'eval' --fcn_model 'save/fcn/fcn_model.pt' --reg_model 'downloads/reg_model.pt' --n_scenes 20

python3 main.py --mode 'fcn' --dataset_dir 'save/ppg-dataset' --epochs 100 --batch_size 5

python3 collect_data.py --singulation_condition --n_samples 30000 --seed 9 #8 #7 #6 #5 #4 #1

4 - edgar
5,6 - iMac (ou-1)
7 - Regan (ou-2)
8,9,10 - Mac



For the Pose-FCN Pose (Paper):
- To simplify learning of the angle θ, we account for
different pushing directions by rotating the input heightmap into
k = 16 discrete orientations (different multiplies of 22.5◦) and
then consider only horizontal pushes to the right

For the Aperture-CNN
- The depth heightmap I is first rotated according to optimal angle θ∗ and then cropped with
respect to optimal initial position p∗ to form a 64 × 64 map. The
optimal p∗ and θ∗ are produced by the Pose-FCN module.