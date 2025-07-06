# 🌿 Leaf Disease Classification with Vanilla CNN

> ⚗️ **Side Project** | A hands-on experiment to benchmark the capabilities of a vanilla CNN on a real-world classification task.

This repository presents a simple yet insightful experiment aimed at testing how well a **vanilla Convolutional Neural Network (CNN)** can perform in the domain of leaf disease classification. The dataset contains **10 classes** of leaf images representing different plant diseases.

Three training setups were tested to explore how training duration and early stopping affect the model's performance. The goal: understand the baseline potential of a pure CNN architecture—**no transfer learning, no complex tricks.**

---

## 🧠 Model Architecture

This minimal CNN was designed for simplicity and speed:

```
Input (128x128x3)
└── Conv2D (32 filters) + MaxPooling2D
└── Conv2D (16 filters) + MaxPooling2D
└── Conv2D (8 filters)  + MaxPooling2D
└── Flatten
└── Dense (128 units) + Dropout
└── Dense (10 units - Softmax)
```

| Layer               | Output Shape        | Parameters |
|---------------------|---------------------|------------|
| Conv2D (32)         | (126, 126, 32)      | 896        |
| MaxPooling2D        | (63, 63, 32)        | 0          |
| Conv2D (16)         | (61, 61, 16)        | 4,624      |
| MaxPooling2D        | (30, 30, 16)        | 0          |
| Conv2D (8)          | (28, 28, 8)         | 1,160      |
| MaxPooling2D        | (14, 14, 8)         | 0          |
| Flatten             | (1568)              | 0          |
| Dense (128)         | (128)               | 200,832    |
| Dropout             | (128)               | 0          |
| Dense (10 - output) | (10)                | 1,290      |
| **Total Params**    |                     | **208,802**|

---

## 🧪 Training Experiments

To test model generalization, the CNN was trained under three different regimes:

| **Experiment**             | **Epochs** | **EarlyStopping** | **Train Accuracy** | **Val Accuracy** | **Val Loss** |
|---------------------------|------------|-------------------|--------------------|------------------|--------------|
| Short Run                 | 10         | ❌                | 84.89%             | 82.20%           | 0.5900       |
| Balanced Run              | 30         | ❌                | 93.34%             | 86.20%           | 0.5065       |
| Long Run (EarlyStopped)   | 100        | ✅ `patience=2`   | 84.07%             | 82.00%           | 0.5568       |

### 🔍 Key Insights
- **30 epochs** gave the best performance, suggesting that moderate training helps the network converge meaningfully.
- **EarlyStopping** prevented overfitting but also slightly undercut final performance.
- **Short runs** are quick but sacrifice validation accuracy.

---

## 🚀 Getting Started

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/leaf-disease-cnn.git
   cd leaf-disease-cnn
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model:**
   ```bash
   python train_model.py
   ```

4. **Make predictions:**
   ```bash
   python predict.py --image sample_leaf.jpg
   ```

> ⚙️ Training parameters (epochs, early stopping, batch size, etc.) can be customized in `train_model.py`

---

## 📁 Project Structure

| File/Folder        | Description                             |
|--------------------|-----------------------------------------|
| `model.h5`         | Trained model weights                   |
| `train_model.py`   | Script to train the CNN model           |
| `predict.py`       | Run predictions on new leaf images      |
| `requirements.txt` | Python dependencies                     |
| `README.md`        | Project documentation                   |

---

## 🌱 Future Experiments

- Add **data augmentation** to improve generalization
- Compare with **transfer learning** using MobileNet or ResNet
- Use **confusion matrix** or Grad-CAM for visual explainability

---

## 📚 License

MIT License — feel free to use, modify, and share.

---
