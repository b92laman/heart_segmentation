from src.dataset import get_dataset
from src.model import get_model
import torch

def main():

    print("Loading dataset...")
    ds = get_dataset("data")

    print("Dataset size:", len(ds))

    sample = ds[0]
    print("Image shape:", sample["image"].shape)
    print("Label shape:", sample["label"].shape)

    print("Loading model...")
    model = get_model()

    print("Model ready ✔")

if __name__ == "__main__":
    main()