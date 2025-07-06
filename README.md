# 🌿 Leaf Disease Classification: Vanilla CNN vs DenseNet121

> ⚗️ **Side Project** | A hands-on experiment comparing a custom vanilla CNN to a pre-trained DenseNet121 model for plant disease classification.

This repository documents an experiment designed to evaluate how a **vanilla Convolutional Neural Network (CNN)** compares to a more advanced **DenseNet121** architecture (pretrained on ImageNet) for the task of classifying leaf diseases across **10 distinct classes**.

By isolating a basic CNN and training it from scratch, this side project aims to set a performance baseline and contrast it against the transfer learning capabilities of DenseNet121.

---

## 🧠 Vanilla CNN Architecture

The custom CNN was designed to be lightweight, minimal, and interpretable:

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

## 🧪 CNN Training Experiments

To evaluate the vanilla CNN, three training strategies were applied:

| **Experiment**             | **Epochs** | **EarlyStopping** | **Train Accuracy** | **Val Accuracy** | **Val Loss** |
|---------------------------|------------|-------------------|--------------------|------------------|--------------|
| Short Run                 | 10         | ❌                | 84.89%             | 82.20%           | 0.5900       |
| Balanced Run              | 30         | ❌                | 93.34%             | 86.20%           | 0.5065       |
| Long Run (EarlyStopped)   | 100        | ✅ `patience=2`   | 84.07%             | 82.00%           | 0.5568       |

### 🔍 Key Takeaways
- The **30-epoch** configuration performed best, balancing training and validation metrics.
- **EarlyStopping** helped reduce overfitting but slightly impacted generalization.
- The **vanilla CNN** model demonstrates promising results for a basic architecture trained from scratch.

---

## 🆚 CNN vs DenseNet121 (Motivation)

This experiment was conducted to **benchmark the effectiveness of a custom CNN** against a **DenseNet121-based model** trained previously on the same dataset.

| Model           | Validation Accuracy | Notes                        |
|----------------|---------------------|------------------------------|
| **DenseNet121** | 96%                 | Achieved higher accuracy, used pretrained weights and deeper layers |
| **Vanilla CNN** | 86.20%              | Faster, simpler, no transfer learning |

The comparison aims to answer:
- Can a lightweight CNN get close to a heavyweight pretrained model?
- Is transfer learning always worth it for small to medium datasets?

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
| `model.h5`         | Trained CNN model weights               |
| `train_model.py`   | Script to train the CNN model           |
| `predict.py`       | Run predictions on new leaf images      |
| `requirements.txt` | Python dependencies                     |
| `README.md`        | Project documentation                   |

---

## 🌱 Future Experiments

- Include **training graphs** (loss/accuracy vs epochs)
- Add **DenseNet121 training logs** for side-by-side comparison
- Use **Grad-CAM** for visual explainability
- Try **data augmentation** for improved CNN generalization

---

## 📚 License

MIT License — feel free to use, modify, and share.

---
