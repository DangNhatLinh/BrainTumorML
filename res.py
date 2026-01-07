import os
import torch
import matplotlib.pyplot as plt

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    ToTensord,
)
from monai.networks.nets import UNet
from monai.transforms import ResizeWithPadOrCropd

TARGET_SHAPE = (128, 128, 128)
DATA_DIR = "data/BrainTumor_data"
CKPT_PATH = "best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIS_CHANNEL = 0
OUT_PATH = "overlay.png"

images = sorted([os.path.join(DATA_DIR, "imagesTr", f) for f in os.listdir(os.path.join(DATA_DIR, "imagesTr"))])
labels = sorted([os.path.join(DATA_DIR, "labelsTr", f) for f in os.listdir(os.path.join(DATA_DIR, "labelsTr"))])
data = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

val_data = data[-10:]
one_case = [val_data[0]]

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=0, a_max=300,
        b_min=0.0, b_max=1.0,
        clip=True,
    ),
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=TARGET_SHAPE),
    ToTensord(keys=["image", "label"]),
])

ds = Dataset(one_case, transform=transforms)
loader = DataLoader(ds, batch_size=1, shuffle=False)

model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
).to(DEVICE)

model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

with torch.no_grad():
    batch = next(iter(loader))
    x = batch["image"].to(DEVICE) # [1, 4, H, W, D]
    y = batch["label"].to(DEVICE) #[1, 1, H, W, D]

    logits = model(x) # [1, 2, H, W, D]
    pred = torch.argmax(logits, dim=1, keepdim=True)  #[1, 1, H, W, D]

x = x.cpu()[0] # [4, H, W, D]
y = y.cpu()[0, 0] # [H, W, D]
pred = pred.cpu()[0, 0]# [H, W, D]

tumor_per_slice = y.sum(dim=(0, 1))  # sum over H,W -> [D]
z = int(torch.argmax(tumor_per_slice).item())

img_slice = x[VIS_CHANNEL, :, :, z]
gt_slice = y[:, :, z]
pred_slice = pred[:, :, z]

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img_slice, cmap="gray")
plt.imshow(gt_slice, alpha=0.4)
plt.title("Ground Truth Overlay")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_slice, cmap="gray")
plt.imshow(pred_slice, alpha=0.4)
plt.title("Prediction Overlay")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
print(f"Saved: {OUT_PATH}")
