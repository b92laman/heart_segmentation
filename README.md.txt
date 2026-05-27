# Heart Segmentation using MONAI and PyTorch

This project implements a 3D medical image segmentation pipeline using a U-Net architecture to segment cardiac structures from MRI/CT volumes.

Built with PyTorch and MONAI for medical imaging workflows.

---

## 🚀 Features

- 3D medical image loading (.nii.gz)
- Dataset preprocessing and augmentation
- 3D U-Net model
- Training pipeline
- Inference with visualization
- 3D GIF generation of slices
- Prediction overlay visualization

---

## 🧠 Technologies Used

- PyTorch
- MONAI
- NumPy
- Matplotlib
- ImageIO
- OpenCV

---

## 📁 Project Structure

```text
heart_segmentation/
│
├── src/
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ ├── inference.py
│ ├── utils.py
│ └── test.py
│
├── data/
│ ├── images/
│ └── labels/
│
├── results/
│ ├── results.png
│ └── heart.gif
│
├── main.py
├── requirements.txt
└── README.md
```
---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/heart_segmentation.git
cd heart_segmentation

pip install -r requirements.txt
```

## 🚀 How to Run

### 1. Train the model

```bash
python src/train.py
```

### 2. Run inference

```bash
python src/inference.py
```

---

## 📊 Results

Example output:

🔹 Slice prediction

Saved in:

results/results.png

🔹 3D visualization (GIF)

Saved in:

results/heart.gif

---

## 🧠 Goal

This project demonstrates a full medical imaging pipeline using deep learning for 3D segmentation of cardiac structures.

It is based on U-Net architectures widely used in medical image analysis.

---

## 📌 Notes

- Data must be in .nii.gz format
- Ensure correct pairing of images and labels
- GPU not required but recommended

--- 

## ✍️ Author

Nerea Lázaro
