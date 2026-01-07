import os
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    ToTensord,
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import ResizeWithPadOrCropd
from monai.transforms import Lambdad # since labels aren't binary i'm scared so i'm putting here just in case

TARGET_SHAPE = (128, 128, 128)

DATA_DIR = "data/BrainTumor_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 1
LR = 1e-4

images = sorted([os.path.join(DATA_DIR, "imagesTr", f) for f in os.listdir(os.path.join(DATA_DIR, "imagesTr"))])
labels = sorted([os.path.join(DATA_DIR, "labelsTr", f) for f in os.listdir(os.path.join(DATA_DIR, "labelsTr"))])
data = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

train_data = data[:-10]
val_data = data[-10:]

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=0, a_max=300, b_min=0.0, b_max=1.0, clip=True,
        ),
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=TARGET_SHAPE),
    ToTensord(keys=["image", "label"]),
])

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=300, b_min=0.0, b_max=1.0, clip=True),

    Lambdad(keys=["label"], func=lambda x: (x > 0).astype(x.dtype)),
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=TARGET_SHAPE),
    ToTensord(keys=["image", "label"]),
])

train_ds = Dataset(train_data, transform = transforms)
val_ds = Dataset(val_data, transform = transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
).to(DEVICE)

loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
dice_metric = DiceMetric(include_background=False, reduction="mean")

best_dice = -1.0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        x = batch["image"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    model.eval()
    dice_metric.reset()

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            logits = model(x)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            dice_metric(preds, y)

    dice = dice_metric.aggregate().item()

    if dice > best_dice:
        best_dice = dice
        torch.save(model.state_dict(), "best_model.pt")

    print(f"Epoch {epoch+1}/{EPOCHS} | loss={epoch_loss:.4f} | val_dice={dice:.4f}")

print("Training done.")