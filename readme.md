# Website Interface
![alt text](<Screenshot 2025-06-23 001849.png>)

# Detecting Disease
![alt text](<Screenshot 2025-06-23 001921.png>)

# About Page
![alt text](<Screenshot 2025-06-23 001943.png>)

# 🩺Lungs Disease Detection Framework

A deep learning-based framework to detect Lungs diseases from chest X-ray images using Convolutional Neural Networks (CNN).

# Before Running , download and store dataset in raw folder 


# 🩺Lungs Disease Detection Framework

A deep learning-based framework to detect Lungs diseases from chest X-ray images using Convolutional Neural Networks (CNN).

## 📁 Project Structure

```
disease_detection_framework/
├── data/
│   ├── raw/                # Raw dataset
│   └── processed/          # Preprocessed data
│       ├── train/
│       └── val/
|
├── notebooks/
│   ├── EDA.ipynb           # Data exploration
│   ├── training.ipynb      # Model training
│   └── test.ipynb          # Model testing
|
├── src/
│   ├── data_loader.py      # Data pipeline
│   ├── model.py            # CNN model definition
│   └── evaluate1.py        # Evaluation utilities
|   └── evaluate.py         # Evaluation utilities
|
├── train.py                # Training script
|
├── models/                 # Saved models
|
├── outputs/                # Graphs and visualizations
|
├── reports/                # Evaluation reports
|
├── requirements.txt        # Required packages
|
├── README.md               # This file
|
├── templates/              
│   ├── index.html/         #Home Page Of Prediction deployment
│   └── about.html/         #About section 
|
└── app.py                  # Optional web interface
```

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Download and extract dataset** into `data/raw/`
4. **Run the split script** to prepare training/validation sets
5. **Train the model**
```bash
python train.py
```
6. **Test the model** using `test.ipynb`

## 📊 Model
- Initial version uses a custom CNN built with Keras.
- Later versions can use transfer learning (e.g., ResNet, DenseNet).

## 🧪 Evaluation
- Accuracy, precision, recall, and confusion matrix metrics are saved under `reports/` and visualized in `outputs/plots/`.

## 🌐 Deployment
- Use `app.py` with **Streamlit** or **Flask** to build a web-based prediction interface.

## 📌 Notes
- Make sure your data folders follow the expected format (`train/class_name/` and `val/class_name/`).
- Use the provided modular codebase for clean experimentation and scalability.

---

Feel free to contribute or raise issues!

---

