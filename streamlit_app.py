import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Title
st.title("🎬 Movie Sentiment Analysis App")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("IMDB Dataset.csv")
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    return data

data = load_data()

X = data['review']
y = data['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tokenization
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Build model
@st.cache_resource
def train_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train_pad,
        y_train,
        epochs=2,
        batch_size=64,
        validation_split=0.2,
        verbose=0
    )
    return model

model = train_model()

# Prediction function
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)[0][0]
    
    return "Positive 😊" if pred > 0.5 else "Negative 😡"

# UI Input
user_input = st.text_area("Enter your movie review:")

if st.button("Predict"):
    if user_input.strip() != "":
        result = predict_sentiment(user_input)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text!")