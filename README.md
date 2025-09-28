# 🍅 Tomato Leaf Disease Detection using CNN

This project uses a Convolutional Neural Network (CNN) to classify tomato leaf images into 10 disease categories and 1 healthy class. It leverages a curated image dataset to build a robust deep learning model for plant disease diagnosis.

---

## 📂 Dataset Overview

- **Source**: [Kaggle – Tomato Leaf Disease Detection Dataset](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)
- **Total Images**: ~18,000+
- **Classes**:
  - Bacterial Spot
  - Early Blight
  - Late Blight
  - Leaf Mold
  - Septoria Leaf Spot
  - Spider Mites (Two-spotted)
  - Target Spot
  - Tomato Mosaic Virus
  - Tomato Yellow Leaf Curl Virus
  - Tomato Healthy

- **Structure**:
- 
Tomato_Leaf_Dataset/

├── train/

│ ├── Tomato___Bacterial_spot/

│ ├── Tomato___Early_blight/

│ └── ...

└── val/

├── Tomato___Bacterial_spot/

├── Tomato___Early_blight/

└── ...


- **Format**: JPEG images, 256×256 resolution

---

## 🧠 Model Architecture

- **Framework**: TensorFlow / Keras
- **Model**: Custom CNN with:
- Conv2D + MaxPooling layers
- Dropout for regularization
- Dense layers with softmax output
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam

- **Format**: JPEG images, 256×256 resolution

---

## 🧠 Model Architecture

- **Framework**: TensorFlow / Keras
- **Model**: Custom CNN with:
- Conv2D + MaxPooling layers
- Dropout for regularization
- Dense layers with softmax output
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam

## 🚀 How to Run

 Clone the repo:

 git clone https://github.com/AK-Jeevan/Tomato-leaf-disease-detection-using-CNN.git
 cd Tomato-leaf-disease-detection-using-CNN

## 🧾 License
This project is released under the MIT License. Dataset is under CC0: Public Domain.
