# Skin-Lesion-Images-for-Melanoma-Classification
The model aims to distinguish between benign and malignant cases, assisting dermatologists in clinical diagnosis and contributing to improved early detection rates.


This project compares two different deep learning approaches for multi-class classification of skin lesion images:
- A **custom Convolutional Neural Network (CNN) with a focal loss**, trained from scratch.
- A **Transfer Learning model using EfficientNetB5 pre-trained on ImageNet**.

---

## 📁 Dataset

We use the **ISIC 2019 dataset**, which includes:
- High-resolution images of skin lesions.
- Metadata (age, sex, anatomical site).
- Multi-label ground truth for different diagnoses such as: `MEL`, `NV`, `BCC`, `AK`, `BKL`, `DF`, `VASC`, `SCC`.

---

## 🔍 Objective

The main goals of this project are:
- To evaluate the performance of a custom CNN vs. a transfer learning model.
- To understand how well each model classifies images into the 8 skin lesion categories.
- To analyze the challenges with imbalanced datasets and apply focal loss or class weights.
- To visualize model learning curves and classification performance.

---

## 🧪 Model 1: RNN (Custom CNN)

### Architecture
- Simple CNN with 2 convolutional layers followed by a dense classifier.
- Trained from scratch.
- Uses **Focal Loss** to handle class imbalance.

### Results
- **Validation Accuracy: ~50.8% %**
- The model showed **no significant learning** after the first epoch.
- LR scheduler reduced the learning rate, but no gain in performance.

### Interpretation
- The model lacks capacity to learn complex features.
- Despite data augmentation and class balancing, its shallow architecture is insufficient.
- Likely overfitting to frequent classes or stuck in local minima.

### Suggestions for Improvement
- Use deeper CNN architectures (e.g., ResNet, MobileNet).
- Try pretrained models (i.e., Transfer Learning).
- Experiment with larger batch sizes, learning rate schedules, or dropout tuning.

---

## 🧠 Model 2: Transfer Learning (EfficientNetB5)

### Architecture
- EfficientNetB5 base pretrained on ImageNet.
- Top layers: GlobalAveragePooling → Dense(512) → Dropout → Output layer.
- Only top layers trained initially; fine-tuning enabled partially.

### Training
- Uses **Sparse Categorical Crossentropy** and **SGD optimizer**.
- No early stopping or LR scheduling used (could be added for further tuning).

### Results
- **Validation Accuracy: ~80%**
- Significant learning and convergence observed over epochs.

### Why This Model Performed Better
- EfficientNetB5 brings rich, general-purpose features from ImageNet pretraining.
- Fine-tuning adapts the network to skin lesion data.
- Proper use of data augmentation and class weighting helped avoid overfitting.
- Deeper architecture captures nuanced patterns in lesion images.

---

## 📊 Visualizations

Both models include:
- Accuracy & Loss plots.
- Confusion Matrix.
- Classification Report (precision, recall, F1).

---

## ✅ Conclusion

| Model           | Validation Accuracy | Notes                            |
|----------------|---------------------|----------------------------------|
| Custom CNN (RNN) | ~50.8%              | Underfitting, ineffective learning |
| EfficientNetB5  | ~80%                | Strong performance with transfer learning |

- Transfer Learning is far superior for this image classification task.
- Focal loss in the custom CNN helps a bit with class imbalance but can't fix architectural limitations.
- Pretrained models are crucial for tasks with limited or complex medical data.

---
