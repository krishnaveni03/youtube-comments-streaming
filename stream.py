import re
import random
import streamlit as st
import googleapiclient.discovery
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os

# Get the absolute path to the model file (adjust as needed)
model_file_path = os.path.join(os.path.dirname(__file__), 'sentiment_analysis.h5')  # Assuming model is in the same directory

# Load the Keras model (with error handling)
try:
  model = keras.models.load_model(model_file_path)
except Exception as e:
  st.error(f"Failed to load the model file: {e}")
  model = None  # Set model to None if loading fails

# Tokenizer configuration (must match the one used for training)
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)

# Initialize the YouTube Data API client
DEVELOPER_KEY = 'AIzaSyDub1h7J9kgxhRTZaWHi7HH-3Nr5DMzWYA'
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=DEVELOPER_KEY)

def extract_video_id(video_link):
    # ... (code remains the same)

def get_random_comments(video_id):
    # ... (code remains the same)

def predict_sentiment(comment, tokenizer, model=model, threshold=0.5):
    if model is None:  # Check if model is loaded
        return "Model not loaded"

    tokenizer.fit_on_texts([comment])
    sequences = tokenizer.texts_to_sequences([comment])
    X = pad_sequences(sequences, maxlen=100)
    prediction = model.predict(X)
    sentiment = "positive" if prediction[0][0] > threshold else "negative"
    return sentiment

def main():
    st.title('Sentiment Analysis on YouTube Comments')

    video_link = st.text_input('Enter YouTube video link:')

    if video_link:
        video_id = extract_video_id(video_link)
        if video_id:
            comments = get_random_comments(video_id)
            selected_comment = st.selectbox('Select a comment:', comments)
            if selected_comment:
                try:
                    sentiment = predict_sentiment(selected_comment, tokenizer, model=model)
                    st.write('Selected Comment:', selected_comment)
                    st.write('Predicted Sentiment:', sentiment)
                except Exception as e:
                    st.error(f"Error predicting sentiment: {e}")
        else:
            st.error("Invalid YouTube video link format.")

if __name__ == '__main__':
  main()
