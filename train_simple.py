

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Hyperparameters
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
BATCH_SIZE = 32
EPOCHS = 10

def get_dataset_path():

    if os.path.exists('labels.csv'):
        return '.'
    else:
        print("\n[!] Could not find 'labels.csv' in this exact folder.")
        folder = input("Please type or paste the exact path to the folder containing labels.csv and images: ").strip()
        # Remove any quotes from copy-pasting 
        folder = folder.strip('\"').strip('\'')
        return folder

def load_data(dataset_path):
    
    labels_file = os.path.join(dataset_path, 'labels.csv')
    
    img_dir_1 = os.path.join(dataset_path, 'images')
    img_dir_2 = os.path.join(dataset_path, 'images', 'images')
    
    if os.path.exists(img_dir_2):
        images_dir = img_dir_2
    else:
        images_dir = img_dir_1

    print(f"Loading labels from: {labels_file}")
    df = pd.read_csv(labels_file)

    perform_eda(df, images_dir)

    # Some labels might be uppercase/lowercase, standardize them
    df['overall_sentiment'] = df['overall_sentiment'].str.lower()
    
    # Create a mapping for raw string labels to numbers (0, 1, 2, ...)
    unique_labels = df['overall_sentiment'].unique()
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Found {num_classes} classes: {unique_labels}")

    images = []
    labels = []

    print("Loading and resizing images... (this may take a moment depending on the dataset size)")
    
    count = 0
    for index, row in df.iterrows():
        img_name = str(row['image_name'])
        # Add .jpg if not there (adjust based on your dataset)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name += '.jpg'
            
        img_path = os.path.join(images_dir, img_name)
        
        if os.path.exists(img_path):
            try:
                # Open, convert to RGB, and resize to 64x64
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                images.append(np.array(img))
                labels.append(label_to_index[row['overall_sentiment']])
                count += 1
            except Exception as e:
                pass
                

    X = np.array(images, dtype='float32') / 255.0  # Normalize to 0-1
    y = to_categorical(np.array(labels), num_classes=num_classes)
    
    print(f"Successfully loaded {len(X)} images.")
    return X, y, label_to_index

def perform_eda(df, images_dir):
    
    print("\n" + "="*40)
    print("--- Starting Exploratory Data Analysis (EDA) ---")
    print("="*40)
    
    # 1. Basic Dataframe info
    print("\nDataset Head:")
    print(df.head())
    print("\nDataset Shape:", df.shape)
    print("\nDataset Columns:", df.columns.tolist())
    print("\nMissing Values Check:")
    print(df.isnull().sum())
    print("\nDataset Info:")
    df.info()

    # 2. Visualize Missing Values Heatmap
    print("\nGenerating Missing Values Heatmap...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()

    # 3. Visualize Emotion Distribution
    print("Generating Sentiment Distribution Plot...")
    plt.figure(figsize=(10, 6))
    # Standardize temporarily for the plot
    plot_df = df.copy()
    plot_df['overall_sentiment'] = plot_df['overall_sentiment'].astype(str).str.lower()
    sns.countplot(x="overall_sentiment", data=plot_df, hue="overall_sentiment", palette="viridis", legend=False)
    plt.title("Distribution of Sentiments in Memes")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    # 4. Display a Sample of Images along with their sentiments
    print("Generating Sample Images Grid...")
    sample_size = min(6, len(df))
    sample_df = df.sample(sample_size)
    
    plt.figure(figsize=(15, 10))
    # Keep track if we actually plotted any images
    images_plotted = False 
    for i, (index, row) in enumerate(sample_df.iterrows()):
        img_name = str(row['image_name'])
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name += '.jpg'
        img_path = os.path.join(images_dir, img_name)
        
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                plt.subplot(2, 3, i+1)
                plt.imshow(img)
                plt.title(f"Sentiment: {row['overall_sentiment']}")
                plt.axis('off')
                images_plotted = True
            except Exception as e:
                pass

    if images_plotted:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()
        print("Note: Could not load sample images for EDA grid. Check images directory.")
    
    print("="*40)
    print("--- EDA Complete ---")
    print("="*40 + "\n")

def build_simple_cnn(num_classes):

    model = Sequential([
        # Layer 1: Detect basic shapes and lines
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Layer 2: Detect more complex patterns
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Classify
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Helps prevent the model from simply memorizing the training data
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def main():
    # 1. Get the path to dataset
    dataset_path = get_dataset_path()
    
    # 2. Load and preprocess the data
    X, y, label_mapping = load_data(dataset_path)
    
    if len(X) == 0:
        print("Error: No images were loaded. Please check the dataset path.")
        return

    # 3. Split the dataset into Training (80%) and Testing (20%)
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Build the model
    print("Building the simple CNN model...")
    num_classes = len(label_mapping)
    model = build_simple_cnn(num_classes)
    model.summary()
    
    # 5. Train the model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test)
    )
    
    # 6. Evaluate and Save
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
    
    model.save('meme_emotion_model.h5')
    print("Model saved to 'meme_emotion_model.h5'")
    
    # Save the label mapping so we know what prediction numbers mean later
    with open('label_mapping.txt', 'w') as f:
        for label, idx in label_mapping.items():
            f.write(f"{idx}:{label}\n")
    print("Label mapping saved to 'label_mapping.txt'")

if __name__ == "__main__":
    main()
