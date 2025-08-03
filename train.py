# train.py

import os
from src.data_loader import create_data_generators
from src.model import build_cnn_model

def train_model(train_dir, val_dir, model_save_path, img_size=(224, 224), batch_size=32, epochs=10):
    train_gen, val_gen = create_data_generators(train_dir, val_dir, img_size, batch_size)

    model = build_cnn_model(input_shape=(img_size[0], img_size[1], 3), num_classes=train_gen.num_classes)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_dir = 'data/processed/train'
    val_dir = 'data/processed/val'
    model_save_path = 'models/cnn_model.h5'
    
    train_model(train_dir, val_dir, model_save_path)
