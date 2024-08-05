# PRODIGY_ML_03
"SVM-Based Classifier for Cats and Dogs Image Recognition"
This project implements a Support Vector Machine (SVM) classifier to classify images of cats and dogs. The dataset used for this project is sourced from the Kaggle competition Dogs vs. Cats, which includes a large collection of labeled images of cats and dogs.

Dataset
The dataset from Kaggle contains:

Training Data: Images of cats and dogs, with each image labeled according to its category.
Test Data: Unlabeled images for which predictions will be made.
You can download the dataset from here.

Methodology
Data Preparation:

Load Images: Images are loaded from the dataset directories. Each image is read in grayscale to simplify the processing.
Resize and Flatten: Images are resized to a uniform size (64x64 pixels) and flattened into 1D arrays for use with the SVM model.
Feature Extraction:

Standardization: Features are scaled to ensure that the SVM model performs optimally by having all features on a similar scale.
Model Training:

Training SVM: A Support Vector Machine with a linear kernel is trained on the preprocessed training images. The model learns to differentiate between the images of cats and dogs.
Model Evaluation:

Validation: The model is evaluated on a validation set, with metrics such as accuracy, precision, recall, and F1-score reported to assess performance.
Prediction:

Testing: The trained model is used to predict labels for the test images. Predictions are output to a CSV file, mapping each image filename to its predicted label ('cat' or 'dog').
Results
The final predictions for the test images are saved in a CSV file, predictions.csv, which includes filenames and predicted labels. This file can be used to assess the model’s performance on unseen data.

This project showcases the application of machine learning for image classification tasks, demonstrating how SVM can be used for binary classification problems in computer vision.
Datasets from https://www.kaggle.com/c/dogs-vs-cats/data





