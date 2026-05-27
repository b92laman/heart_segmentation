from dataset import get_dataset

def main():

    ds = get_dataset("data")

    print("Dataset size:", len(ds))

    if len(ds) == 0:
        print("❌ Dataset vacío")
        return

    sample = ds[0]

    print("Image shape:", sample["image"].shape)
    print("Label shape:", sample["label"].shape)

    print("✔ Dataset OK")

if __name__ == "__main__":
    main()