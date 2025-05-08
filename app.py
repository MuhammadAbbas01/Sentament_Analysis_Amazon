import os
import bz2
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup
nltk.download('stopwords')
nltk.download('wordnet')
sns.set_context('notebook')
sns.set_style('white')
np.random.seed(2024)

# Load data
def load_amazon_reviews(file_path, max_lines=20000):
    data = []
    with bz2.open(file_path, 'rt') as file:
        for i, line in enumerate(file):
            if i >= max_lines: break
            label, text = int(line[9]) - 1, line[10:].strip()
            data.append((label, text))
    return pd.DataFrame(data, columns=["label", "review"])

df = load_amazon_reviews("train.ft.txt.bz2")

# Clean & preprocess
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z']", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = clean_text(text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

df["review"] = df["review"].apply(preprocess_text)

# Tokenization
max_words = 10000
max_len = 150
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df["review"])
X = pad_sequences(tokenizer.texts_to_sequences(df["review"]), maxlen=max_len)
y = np.array(df["label"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Class weights
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(weights))

# Model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Training
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=2
)

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot accuracy/loss
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(loc='lower right'); plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(loc='upper right'); plt.show()

# Predictions
y_pred_probs = model.predict(X_test, batch_size=64)
y_pred_labels = (y_pred_probs > 0.5).astype("int32")

# Metrics with safety check
if len(np.unique(y_test)) == 1:
    print("Warning: Only one class in test labels. Skipping precision/recall/f1 computation.")
else:
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_labels, average='binary')
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:\n", cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix'); plt.show()

# Sample predictions
sample_reviews = [
    "Absolutely terrible product. I hated it!",
    "This is the best thing I've ever bought. Totally worth it!",
    "Not bad, but definitely could be better.",
    "I love how this works. Great job!",
    "Meh. It's okay, nothing special."
]
sample_cleaned = [clean_text(r) for r in sample_reviews]
sample_seq = tokenizer.texts_to_sequences(sample_cleaned)
sample_pad = pad_sequences(sample_seq, maxlen=max_len)
sample_probs = model.predict(sample_pad)

# Best threshold
val_probs = model.predict(X_test)
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, val_probs)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
best_thresh = thresholds[np.argmax(f1_scores)]
print(f"Best threshold: {best_thresh:.4f}")

# Output predictions
for i, review in enumerate(sample_reviews):
    label = "Positive" if sample_probs[i] > best_thresh else "Negative"
    print(f"Review: {review} \n→ Prediction: {label} (Confidence: {sample_probs[i][0]:.2f})\n")
