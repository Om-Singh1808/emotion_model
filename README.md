# Meme Emotion Classifier

A CNN-based deep learning model that analyzes meme images and classifies the emotion/sentiment they convey.

## Overview

This project uses a simple Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify memes into **5 sentiment categories**:

| Index | Emotion        |
|-------|----------------|
| 0     | Very Positive  |
| 1     | Positive       |
| 2     | Neutral        |
| 3     | Negative       |
| 4     | Very Negative  |

## Project Structure

```
├── train_simple.py      # Training script with EDA & CNN model
├── test_simple.py       # Inference script for single image prediction
├── reference.py         # Reference EDA code (Titanic dataset example)
├── label_mapping.txt    # Mapping of class indices to emotion labels
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Requirements

```
tensorflow
numpy
pandas
matplotlib
seaborn
Pillow
scikit-learn
```

Install with:
```bash
pip install tensorflow numpy pandas matplotlib seaborn Pillow scikit-learn
```

## Dataset

This project uses the [6992 Labeled Meme Images Dataset](https://www.kaggle.com/datasets) from Kaggle. Download and place the dataset in the project root:
- `labels.csv` — CSV file with image names and sentiment labels
- `images/` — Directory containing the meme images

> **Note:** The dataset is not included in this repository due to size constraints.

## Usage

### Training
```bash
python train_simple.py
```
This will:
1. Perform Exploratory Data Analysis (EDA) with visualizations
2. Load and preprocess all images (resized to 64×64)
3. Train a 2-layer CNN for 10 epochs
4. Save the trained model as `meme_emotion_model.h5`
5. Save the label mapping to `label_mapping.txt`

### Prediction
```bash
python test_simple.py <path_to_meme_image.jpg>
```
This loads the trained model and outputs the predicted emotion with confidence score.

## Model Architecture

```
Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(5, Softmax)
```

## License

This project is for educational purposes.
