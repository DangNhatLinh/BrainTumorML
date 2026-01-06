# 3D Brain Tumor Segmentation with U-Net

This project implements a **3D convolutional neural network (U-Net)** for volumetric brain tumor segmentation using multi-modal MRI scans. The goal is to build a minimal but correct baseline that demonstrates end-to-end handling of 3D medical imaging data, including preprocessing, training, evaluation, and visualization.

## Dataset
The project uses the **Medical Segmentation Decathlon – Task01_BrainTumour** dataset.

Each training sample consists of a 3D MRI volume with four modalities:
- T1
- T1ce
- T2
- FLAIR

Ground-truth labels provide voxel-level tumor annotations.

### Download
Download the dataset from:
http://medicaldecathlon.com/downloads/Task01_BrainTumour.tar

Extract it into:
data/BrainTumor_data/
imagesTr/
labelsTr/

> The dataset is **not included** in this repository and is ignored via `.gitignore`.

## Environment
Recommended setup using Conda:
```bash
conda create -n brain3d python=3.10
conda activate brain3d
pip install torch monai nibabel matplotlib tqdm "numpy<2.0" "scipy<1.14"
```
## Training
Run training with:
```bash
python train.py
```
The model trains for a fixed number of epochs and saves the best-performing checkpoint based on validation Dice score.

## Inference & Visualization
To visualize predictions on a single validation case:
```bash
python predict_one.py
```
This generates an overlay image comparing ground-truth and predicted tumor masks.

## Notes
- The model performs binary segmentation (tumor vs background).
- This project focuses on correctness and clarity rather than state-of-the-art performance.
- The resulting segmentation captures tumor location and extent but produces coarse boundaries, typical of early baselines.
