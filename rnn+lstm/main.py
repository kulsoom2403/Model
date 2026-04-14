import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense

data = pd.read_csv("IMDB Dataset.csv")

print(data.head())
print(data.shape)

data['sentiment'] = data['sentiment'].map({
    'positive': 1,
    'negative': 0
})

X = data['review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vocab_size = 10000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 200

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

rnn_model = Sequential()

rnn_model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len))
rnn_model.add(SimpleRNN(32))
rnn_model.add(Dense(1, activation='sigmoid'))

rnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training RNN...")

rnn_model.fit(
    X_train_pad,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

loss, acc = rnn_model.evaluate(X_test_pad, y_test)
print("RNN Accuracy:", acc)

# Build LSTM Model (Better)

lstm_model = Sequential()

lstm_model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len))
lstm_model.add(LSTM(64))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train LSTM
print("Training LSTM...")

lstm_model.fit(
    X_train_pad,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)
# Evaluate LSTM
loss, acc = lstm_model.evaluate(X_test_pad, y_test)
print("LSTM Accuracy:", acc)

# Test Your Model

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = lstm_model.predict(padded)[0][0]
    
    if pred > 0.5:
        return "Positive 😊"
    else:
        return "Negative 😡"


print(predict_sentiment("This movie was amazing"))
print(predict_sentiment("Worst movie ever"))