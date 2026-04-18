"""
Simple Meme Emotion Predictor
Loads the trained model and predicts the emotion of a single meme.
"""

import os
import sys
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

def load_label_mapping():
    """Loads the label mapping created during training."""
    mapping = {}
    with open('label_mapping.txt', 'r') as f:
        for line in f:
            idx, label = line.strip().split(':')
            mapping[int(idx)] = label
    return mapping

def predict_emotion(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return

    # Load the trained model
    if not os.path.exists('meme_emotion_model.h5'):
        print("Error: Model not found. Please run train_simple.py first!")
        return
        
    print("Loading model...")
    model = load_model('meme_emotion_model.h5')
    label_mapping = load_label_mapping()
    
    # Preprocess the image
    print(f"Analyzing image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    
    # Neural Networks expect a batch of images, so we add an extra dimension
    # (1 image, 64 height, 64 width, 3 channels)
    img_array = np.array(img, dtype='float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    emotion = label_mapping[predicted_class_index]
    
    print("-" * 30)
    print(f"Predicted Emotion: {emotion.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_simple.py <path_to_image.jpg>")
    else:
        predict_emotion(sys.argv[1])
