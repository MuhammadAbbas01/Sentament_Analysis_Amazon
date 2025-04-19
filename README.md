# Sentiment Analysis on Amazon Product Reviews

This project performs sentiment analysis on Amazon product reviews using natural language processing (NLP) and machine learning techniques. The model is trained to predict whether a review is "Positive" or "Negative" based on the text of the review.

## Overview

The goal of this project is to analyze Amazon product reviews and predict the sentiment of the review (positive or negative). The model is trained using a dataset of reviews, cleaned and preprocessed to ensure optimal performance. The approach uses a neural network architecture with word embeddings to make predictions.

## Key Steps

1. **Data Collection and Preprocessing**: The reviews are cleaned, tokenized, and converted into sequences for model input.
2. **Model Training**: A deep learning model is built and trained using Keras/TensorFlow.
3. **Evaluation**: The model is evaluated on a test dataset to measure its performance.
4. **Sentiment Prediction**: The model is used to predict sentiment on new, unseen reviews.

## Model Usage

To predict sentiment on new reviews, follow these steps:

```python
# Example review
new_review = ["I don't love this product! It's not amazing."]
new_review_cleaned = [clean_text(review) for review in new_review]

# Tokenize and pad the review
new_review_seq = tokenizer.texts_to_sequences(new_review_cleaned)
new_review_pad = pad_sequences(new_review_seq, maxlen=max_len, padding='post', truncating='post')

# Predict sentiment
prediction_prob = model.predict(new_review_pad)
prediction = "Positive" if prediction_prob > optimal_threshold else "Negative"

print(f"Prediction Probability: {prediction_prob[0]:.4f}")
print(f"Prediction: {prediction}")
