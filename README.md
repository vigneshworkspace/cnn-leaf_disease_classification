# 🌿 Leaf Disease Classifier: Vanilla CNN vs DenseNet121

> 🎯 **Side Project Goal**: Benchmark a simple, self-built **vanilla CNN** against a powerful pretrained **DenseNet121** for multi-class leaf disease detection.

This experiment dives into the question: **"How good is a clean, no-frills CNN from scratch compared to a state-of-the-art pretrained model?"**

Both models were trained on the same dataset of diseased leaf images spanning **10 classes**. This comparison evaluates **model simplicity vs transfer learning power.**

---

## 🧠 Model Architectures

### ✅ Vanilla CNN (Minimal Custom Build)
```
Input (128x128x3)
 ├── Conv2D (32 filters) → ReLU
 ├── MaxPooling2D
 ├── Conv2D (16 filters) → ReLU
 ├── MaxPooling2D
 ├── Conv2D (8 filters)  → ReLU
 ├── MaxPooling2D
 ├── Flatten
 ├── Dense (128 units) → ReLU
 ├── Dropout (rate=0.5)
 └── Dense (10 units → Softmax)
```

Total Parameters: **208,802** — Lightweight and fast.

### 🧪 DenseNet121 (Transfer Learning Baseline)
- **Pretrained** on ImageNet
- **Frozen convolutional base** (in early trials)
- **Custom classifier head** for 10-class softmax output
- Significantly more parameters & depth

---

## 📊 Experimental Comparison

| Strategy                   | Model         | Epochs | EarlyStopping | Val Accuracy | Val Loss  |
|---------------------------|---------------|--------|----------------|--------------|-----------|
| Short Run                 | Vanilla CNN   | 10     | ❌             | 82.20%       | 0.5900    |
| Balanced Run              | Vanilla CNN   | 30     | ❌             | 86.20%       | 0.5065    |
| Long Run + Early Stop     | Vanilla CNN   | 100    | ✅ `p=2`       | 82.00%       | 0.5568    |
| Pretrained Transfer Model | DenseNet121   | ~20    | ✅             | **96.00%**   | 0.0554    |

### 🧠 Key Takeaways
- **Vanilla CNN** gave solid results for a basic model with minimal tuning.
- **30 epochs** was the sweet spot — beyond that, returns diminished.
- **DenseNet121** clearly outperformed, showing the power of deep pretrained features.

---

## 🚀 Run It Yourself

```bash
# 1. Clone the repository
$ git clone https://github.com/vigneshworkspace/leaf-disease-cnn.git
$ cd leaf-disease-cnn

# 2. Install dependencies
$ pip install -r requirements.txt

# 3. Train the CNN
$ python cnn_train.py

# 4. Make a prediction
$ streamlit run streamlit.py
```

Customize model configs in `cnn_train.py` 🎛️

---

## 🧾 Project Structure

| File/Folder        | Description                             |
|--------------------|-----------------------------------------|
| `model.h5`         | Trained vanilla CNN weights             |
| `cnn_train.py`     | CNN training script                     |
| `streamlit.py`     | CLI prediction interface                |
| `requirements.txt` | All needed packages                     |
| `README.md`        | This very documentation                 |

---

## 🔮 What's Next?

- [ ] 📈 Add training & validation plots
- [ ] 🧠 Include DenseNet121 code & logs
- [ ] 🌈 Visualize attention with Grad-cam

---

## 🪪 License

MIT License — fork it, modify it, ship it. 🚀
