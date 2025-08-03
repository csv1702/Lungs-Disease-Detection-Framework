# src/evaluate1.py


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
MODEL_PATH = "models/cnn_model.h5"
TEST_DIR = "data/processed/val"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)

# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict
y_pred_probs = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification report
report = classification_report(y_true, y_pred_classes, target_names=class_labels)
with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
print("\nClassification Report:\n", report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
plt.show()

# Accuracy
accuracy = accuracy_score(y_true, y_pred_classes)
with open(os.path.join(REPORT_DIR, "accuracy.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
print(f"\nAccuracy: {accuracy:.4f}")

# F1 Score (macro and weighted)
f1_macro = f1_score(y_true, y_pred_classes, average='macro')
f1_weighted = f1_score(y_true, y_pred_classes, average='weighted')
with open(os.path.join(REPORT_DIR, "f1_scores.txt"), "w") as f:
    f.write(f"F1 Score (Macro): {f1_macro:.4f}\n")
    f.write(f"F1 Score (Weighted): {f1_weighted:.4f}\n")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")

# Sample prediction visualization
import random
from tensorflow.keras.preprocessing import image

plt.figure(figsize=(12, 12))
for i in range(9):
    idx = random.randint(0, len(y_pred_classes) - 1)
    img_path = test_generator.filepaths[idx]
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    plt.subplot(3, 3, i + 1)
    plt.imshow(img_array)
    plt.axis('off')
    true_label = class_labels[y_true[idx]]
    pred_label = class_labels[y_pred_classes[idx]]
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "sample_predictions.png"))
plt.show()

print("\nEvaluation complete. All reports and visualizations saved in 'reports/' folder.")
