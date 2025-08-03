# src/split_dataset.py

import os
import shutil
import random

def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    classes = os.listdir(source_dir)

    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)

        if not os.path.isdir(class_path):
            continue  # Skip non-folder files

        images = os.listdir(class_path)
        random.shuffle(images)

        train_size = int(len(images) * split_ratio)
        train_images = images[:train_size]
        val_images = images[train_size:]

        # Create class folders inside train and val directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Move files
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)

        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy2(src, dst)

    print(f"Dataset split complete: {split_ratio*100}% train, {(1-split_ratio)*100}% validation.")

if __name__ == "__main__":
    source_dir = "data/raw"           # Original dataset
    train_dir = "data/processed/train"
    val_dir = "data/processed/val"
    split_ratio = 0.8                  # 80% training, 20% validation

    split_dataset(source_dir, train_dir, val_dir, split_ratio)
