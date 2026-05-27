import matplotlib.pyplot as plt

def show_slice(volume, slice_idx=None, title="slice"):
    if slice_idx is None:
        slice_idx = volume.shape[-1] // 2

    plt.figure(figsize=(5,5))
    plt.imshow(volume[:, :, slice_idx], cmap="gray")
    plt.title(f"{title} - slice {slice_idx}")
    plt.axis("off")
    plt.show()