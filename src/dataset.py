import os
from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, ToTensord, ResizeWithPadOrCropd
)

def get_dataset(data_dir):

    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    images = sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    ])

    labels = sorted([
        os.path.join(labels_dir, f)
        for f in os.listdir(labels_dir)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    ])

    data = [{"image": i, "label": l} for i, l in zip(images, labels)]

    transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),

    ResizeWithPadOrCropd(
        keys=["image", "label"],
        spatial_size=(256, 256, 128)
    ),

    ToTensord(keys=["image", "label"])
])

    return Dataset(data=data, transform=transforms)