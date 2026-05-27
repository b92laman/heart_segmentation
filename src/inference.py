import os
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

from dataset import get_dataset
from model import get_model


def create_gif(volume, save_path="results/heart.gif"):

    frames = []

    for i in range(volume.shape[-1]):

        slice_img = volume[:, :, i]

        plt.figure()
        plt.imshow(slice_img, cmap="gray")
        plt.axis("off")

        temp_path = f"temp_{i}.png"
        plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        frames.append(imageio.imread(temp_path))
        os.remove(temp_path)

    imageio.mimsave(save_path, frames, fps=6)

    print("✔ GIF saved:", save_path)


def save_result():

    ds = get_dataset("data")
    model = get_model()

    model.eval()

    sample = ds[0]

    img = sample["image"].unsqueeze(0)
    label = sample["label"]

    with torch.no_grad():
        pred = model(img)

    pred = torch.softmax(pred, dim=1)

    # =========================
    # MIDDLE SLICE VISUALIZATION
    # =========================
    mid = img.shape[-1] // 2

    img_slice = img[0, 0, :, :, mid].cpu().numpy()
    label_slice = label[0, :, :, mid].cpu().numpy()
    pred_slice = pred[0, 1, :, :, mid].cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img_slice, cmap="gray")
    

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(label_slice)
    #plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(pred_slice, cmap="jet")
    #plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/results.png")
    plt.show()

    print("✔ Saved image: results/results.png")

    # =========================
    # GIF (FULL VOLUME)
    # =========================
    create_gif(img[0, 0].cpu().numpy())

if __name__ == "__main__":
    save_result()