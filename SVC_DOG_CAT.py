import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Define directories
train_dir = '/content/drive/MyDrive/Colab Datasets/train.zip')'  # Replace with your train directory path
test_dir = '/content/drive/MyDrive/Colab Datasets/test1.zip'    # Replace with your test directory path

# Load images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize images to a fixed size
            images.append(img.flatten())     # Flatten the image for SVM
            labels.append(label)
    return images, labels

# Load training data
cats, cats_labels = load_images_from_folder(os.path.join(train_dir, 'cats'), 0)
dogs, dogs_labels = load_images_from_folder(os.path.join(train_dir, 'dogs'), 1)

# Combine and split data
X = np.array(cats + dogs)
y = np.array(cats_labels + dogs_labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train SVM
clf = svm.SVC(kernel='linear')  # You can also try 'rbf' or other kernels
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))

# Load and predict test images
def load_test_images(test_folder):
    images = []
    filenames = []
    for filename in os.listdir(test_folder):
        img = cv2.imread(os.path.join(test_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize images to a fixed size
            images.append(img.flatten())     # Flatten the image for SVM
            filenames.append(filename)
    return images, filenames

test_images, test_filenames = load_test_images(test_dir)
test_images = scaler.transform(test_images)
test_predictions = clf.predict(test_images)

# Save predictions to a CSV file
import pandas as pd

results = pd.DataFrame({
    'Filename': test_filenames,
    'Label': ['cat' if pred == 0 else 'dog' for pred in test_predictions]
})

results.to_csv('/content/predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'")
