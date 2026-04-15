import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense





df = pd.read_csv("synthetic_emotions.csv")

print(df.head())
print(df.shape)

# Check missing values
print(df.isnull().sum())

# Features & Labels
texts = df['text']
labels = df['emotion']


encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

print(labels[:10])


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

X = pad_sequences(sequences, maxlen=50)



X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)


model = Sequential()

model.add(Embedding(input_dim=5000, output_dim=64, input_length=50))
model.add(LSTM(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(set(labels)), activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

loss, acc = model.evaluate(X_test, y_test)

print("Accuracy:", acc)