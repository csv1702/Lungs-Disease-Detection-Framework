# Website Interface
<img width="3199" height="1661" alt="Screenshot 2025-06-23 001849" src="https://github.com/user-attachments/assets/1ed1fa0b-2fe5-4b4e-8c49-c1cb0039a6e0" />


# Detecting Disease
<img width="3195" height="1662" alt="Screenshot 2025-06-23 001921" src="https://github.com/user-attachments/assets/d95fe6fb-81d4-41ad-a07e-709bdd3d43e1" />


# About Page
<img width="3197" height="1674" alt="Screenshot 2025-06-23 001943" src="https://github.com/user-attachments/assets/b46981f1-0c08-4ef9-9124-bdd4f79f8772" />


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

