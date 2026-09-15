import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Load Dataset from CSV
data = pd.read_csv("LAB2_T_1.CSV")

sentences = data["sentence"].astype(str).tolist()
categories = data["category"].astype(str).tolist()

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

total_words = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(sentences)

max_length = max(len(seq) for seq in sequences)

X = pad_sequences(sequences, maxlen=max_length, padding="post")

# Encode Labels
label_encoder = LabelEncoder()

y_int = label_encoder.fit_transform(categories)

y = to_categorical(y_int)

num_classes = y.shape[1]

# Build Model: Input -> Hidden -> Output
model = Sequential()

model.add(
    Embedding(total_words, 16, input_length=max_length)
)

model.add(Flatten())

model.add(Dense(16, activation="relu"))

model.add(Dense(num_classes, activation="softmax"))

# Compile Model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train Model
model.fit(X, y, epochs=200, verbose=0)

# Classify New Text
def classify(text):
    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=max_length,
        padding="post"
    )

    prediction = model.predict(padded, verbose=0)

    return label_encoder.inverse_transform(
        [np.argmax(prediction)]
    )[0]

sample = input("Enter a sentence: ")

print("Predicted Category:", classify(sample))
