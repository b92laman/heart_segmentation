import torch
from torch.utils.data import DataLoader
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.networks.layers import Norm

from dataset import get_dataset

# ======================
# DEVICE
# ======================
device = torch.device("cpu")

# ======================
# DATA
# ======================
data = get_dataset("data")
loader = DataLoader(data, batch_size=1, shuffle=True)

# ======================
# MODEL
# ======================
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

# ======================
# LOSS + OPTIMIZER
# ======================
loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ======================
# TRAIN LOOP
# ======================
epochs = 3

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    epoch_loss = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print("Loss:", epoch_loss / len(loader))

print("\nTraining finished ✔")