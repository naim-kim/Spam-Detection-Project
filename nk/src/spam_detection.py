#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle

# 1. Collecting a labeled dataset of emails
# Load the CSV file and rename columns
# Load dataset
csv_file_path = "spam.csv"
df = pd.read_csv(csv_file_path, encoding='latin1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print("Dataset Loaded. Shape: ", df.shape)

# 2. Preprocessing the text
def preprocess_text(text):
    # Example preprocessing: lowercasing, removing punctuation, etc.
    text = text.lower()
    # Add more preprocessing steps as needed
    return text

df['text'] = df['text'].apply(preprocess_text)

# 3. Engineering relevant features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 4. Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Selecting and training the machine learning model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Evaluating the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Deployment (this step is usually beyond a Jupyter notebook and involves integrating the model into an email system for live spam detection)

# Example of how to save the model and vectorizer
with open('spam_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved for deployment.")
