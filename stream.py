import re
import random
import streamlit as st
import googleapiclient.discovery
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os

# Get the absolute path to the model file
model_file_path = os.path.join(os.path.dirname(__file__), 'sentiment_analysis.h5')

# Load the Keras model
try:
    model = keras.models.load_model(model_file_path)
    st.write("Model loaded successfully!")
except OSError as e:
    st.error(f"Failed to load the model file: {e}")
    st.stop()  # Stop execution if model loading fails
except Exception as e:
    st.error(f"Error loading Keras model: {e}")
    st.stop()  # Stop execution if model loading fails

# Tokenizer configuration (must match the one used for training)
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)

# Initialize the YouTube Data API client
DEVELOPER_KEY = 'YOUR_DEVELOPER_KEY'
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=DEVELOPER_KEY)

# Define the remaining functions

def main():
    st.title('Sentiment Analysis on YouTube Comments')

    video_link = st.text_input('Enter YouTube video link:')

    if video_link:
        video_id = extract_video_id(video_link)
        if video_id:
            comments = get_random_comments(video_id)
            selected_comment = st.selectbox('Select a comment:', comments)
            if selected_comment is not None:  # Check if selected_comment is not None
                sentiment = predict_sentiment(selected_comment, model, tokenizer)
                st.write('Selected Comment:', selected_comment)
                st.write('Predicted Sentiment:', sentiment)
            else:
                st.warning("Please select a comment.")
        else:
            st.error("Invalid YouTube video link format.")

if __name__ == '__main__':
    main()
