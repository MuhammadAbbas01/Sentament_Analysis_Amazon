# -*- coding: utf-8 -*-
"""Sentamenat_Analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eGVrHwvVuSnucHFQOODM3kz20QKvbiXz
"""

print("hello world")

from google.colab import files

uploaded = files.upload()

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

!ls -l

!mv 'kaggle (10).json' kaggle.json

import os

# Create the .kaggle directory if it does not exist
os.makedirs('/root/.kaggle', exist_ok=True)

# Move the renamed kaggle.json file to the .kaggle directory
!mv kaggle.json /root/.kaggle/

!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d bittlingmayer/amazonreviews

import os

# Print the current working directory
print(os.getcwd())

!unzip amazonreviews.zip

import pandas as pd
import numpy as np
import re
import nltk
import bz2
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import TextVectorization  # Updated import for TextVectorization
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

nltk.download('stopwords')
nltk.download('wordnet')  # Fixed download from 'stopwords' to 'wordnet' to include lemmatizer

# Suppress warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Set visualization styles
import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')
np.random.seed(2024)

# Check TensorFlow version
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# Load Data (limit to 20000 for speed, increase for performance)
def load_amazon_reviews(file_path, max_lines=20000):
    data = []
    with bz2.open(file_path, 'rt') as file:
        for i, line in enumerate(file):
            if i >= max_lines: break
            label, text = int(line[9]) - 1, line[10:].strip()
            data.append((label, text))
    return pd.DataFrame(data, columns=["label", "review"])

df = load_amazon_reviews("train.ft.txt.bz2")
df.head()

# Clean text
def clean_text(text):
    # Remove URLs & HTML
    text = re.sub(r"http\S+|www\S+|<.*?>", "", text)
    # Keep letters only and convert to lower case
    text = re.sub(r"[^a-zA-Z']", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()          # remove extra spaces

# Lemmatization and Stopword Removal
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Clean the text
    text = clean_text(text)
    # Lemmatize and remove stopwords
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df["review"] = df["review"].apply(preprocess_text)
df.head()

# Load Data (limit to 20000 for speed, increase for performance)
def load_amazon_reviews(file_path, max_lines=20000):
    data = []
    with bz2.open(file_path, 'rt') as file:
        for i, line in enumerate(file):
            if i >= max_lines: break
            label, text = int(line[9]) - 1, line[10:].strip()
            data.append((label, text))
    return pd.DataFrame(data, columns=["label", "review"])

df = load_amazon_reviews("train.ft.txt.bz2")

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|<.*?>", "", text)  # remove URLs & HTML
    text = re.sub(r"[^a-zA-Z']", " ", text.lower())   # keep letters only
    return re.sub(r"\s+", " ", text).strip()          # remove extra spaces

df["review"] = df["review"].apply(clean_text)

# Tokenize
max_words = 10000
max_len = 150

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df["review"])
sequences = tokenizer.texts_to_sequences(df["review"])
X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
y = np.array(df["label"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Class weight for imbalance
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(weights))

# EarlyStopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Model: CNN + LSTM
model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(32)),  # Reduced LSTM size to avoid overfitting
    Dropout(0.5),              # Dropout to prevent overfitting
    Dense(64, activation='relu'),  # Added more dense layers
    Dropout(0.5),              # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Output layer
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[early_stop],  # Use EarlyStopping to prevent overfitting
    verbose=2
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")

import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='upper right')
plt.show()

# Predict on the test set
y_pred = model.predict(X_test, batch_size=64)

# Convert predictions to binary labels (0 or 1)
y_pred_labels = (y_pred > 0.5).astype("int32")

# Evaluate performance on test set
test_accuracy = accuracy_score(y_test, y_pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_labels, average='binary')

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

from sklearn.metrics import confusion_matrix

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Example: Predict sentiment of a new review
new_review = ["I don't love this product! it's not  amazing."]
new_review_cleaned = [clean_text(review) for review in new_review]
new_review_seq = tokenizer.texts_to_sequences(new_review_cleaned)
new_review_pad = pad_sequences(new_review_seq, maxlen=max_len, padding='post', truncating='post')

# Predict sentiment
prediction = model.predict(new_review_pad)
print(f"Prediction: {'Positive' if prediction > 0.5 else 'Negative'}")

# Predict sentiment and check the probability
prediction_prob = model.predict(new_review_pad)
print(f"Prediction Probability: {prediction_prob[0][0]:.4f}")
print(f"Prediction: {'Positive' if prediction_prob > 0.5 else 'Negative'}")

new_review = ["This product is terrible. I hate it and want a refund."]

# Sample reviews for sentiment prediction
test_reviews = [
    "Absolutely terrible product. I hated it!",
    "This is the best thing I've ever bought. Totally worth it!",
    "Not bad, but definitely could be better.",
    "I love how this works. Great job!",
    "Meh. It's okay, nothing special."
]

# Step 1: Preprocess reviews (clean, tokenize, pad)
cleaned = [clean_text(r) for r in test_reviews]
sequences = tokenizer.texts_to_sequences(cleaned)
padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Step 2: Predict sentiment probabilities
predicted_probs = model.predict(padded)

# Step 3: Calculate optimal threshold from validation set (if not done earlier)
from sklearn.metrics import precision_recall_curve

val_probs = model.predict(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, val_probs)
f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1)]

# Step 4: Print predictions using the dynamic threshold
print(f"Optimal Dynamic Threshold: {best_threshold:.4f}\n")

for i, prob in enumerate(predicted_probs):
    sentiment = "Positive" if prob > best_threshold else "Negative"
    print(f"Review {i+1}: {test_reviews[i]}")
    print(f"→ Probability: {prob[0]:.4f}")
    print(f"→ Prediction: {sentiment}\n")
