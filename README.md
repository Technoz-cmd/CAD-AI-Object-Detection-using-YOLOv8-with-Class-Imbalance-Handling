# CAD-AI-Object-Detection-using-YOLOv8-with-Class-Imbalance-Handling
Deep Learning project on CAD-AI object detection using YOLOv8. Focuses on dataset preprocessing, handling class imbalance, and improving model performance through multiple modified training runs.

## 📌 Overview
This project focuses on object detection using a YOLOv8-based model on the CAD-AI dataset. The goal was to detect three classes:
- Abiotic
- Insect
- Disease

The main objective was to improve the performance of a baseline YOLOv8s model by applying preprocessing, handling class imbalance, and training modified models.

---

## 🎯 Objectives
- Improve detection performance over baseline YOLOv8s
- Handle class imbalance in dataset
- Clean and validate dataset
- Train and compare multiple model versions
- Analyze performance using standard metrics

---

## 📊 Dataset
- Format: YOLO object detection format
- Splits:
  - Train: 3788 images
  - Validation: 710 images
  - Test: 238 images

### ⚠️ Key Observation
- Dataset is highly imbalanced
- Insect class is dominant
- Abiotic and Disease are minority classes

---

## 🛠️ Preprocessing
Steps performed:
- Verified image-label pairs
- Removed corrupted data
- Checked empty labels
- Fixed boundary issues in annotations
- Validated YOLO label format
- Visual inspection of bounding boxes

### Result:
- Cleaner dataset
- Improved annotation quality
- More reliable training

---

## 🤖 Model Approach

### 🔹 Baseline Model
- YOLOv8s

### 🔹 Proposed Approach
- Modified YOLO-based training
- Multiple training runs
- Fine-tuning using checkpoints
- Class-aware training (focus on minority classes)

---

## 🔁 Training Runs
1. Baseline YOLOv8s
2. Modified Run 1 (50 epochs)
3. Modified Run 2 (100 epochs, higher LR)
4. Modified Run 3 (lower LR, improved training)
5. Final Run (focused on minority classes)

---

## 📈 Results

| Metric        | Baseline | Final Model |
|--------------|----------|------------|
| F1 Score     | 0.58     | 0.60       |
| Precision    | 1.00     | 1.00       |
| Recall       | 0.81     | 0.82       |
| mAP@0.5      | 0.544    | 0.572      |

### Class-wise AP:
- Abiotic: 0.696 → 0.757
- Insect: 0.473 → 0.478
- Disease: 0.463 → 0.482

---

## 📌 Key Insights
- Preprocessing significantly improved data quality
- Early modifications reduced performance
- Proper tuning improved results gradually
- Final model achieved best balance across all classes
- Minority class performance improved notably

---

## 🧠 Conclusion
This project shows that:
- Clean data is critical for good performance
- Handling class imbalance is essential
- Incremental training and tuning lead to better results

The final model outperformed the baseline and is more balanced for real-world use.

---

## 🚀 Future Work
- Improve insect class detection further
- Reduce background confusion
- Try advanced architectures (YOLOv9 / transformers)
- Apply data augmentation techniques

---

## 👤 Author
Siddharth Khatri  
