
# SENTIMENT ANALYSIS

## AIM

To develop a Deep Learning model using a Neural Network for Sentiment Analysis (Positive /
Negative classification), and to study the role of different **activation functions** (ReLU,
Sigmoid, Tanh) in shaping the network's output.

## PROCEDURE

1. Import the required libraries.
2. Read the labelled dataset (200 rows) from the CSV file `lab3_sentiment_dataset.csv` using pandas.
3. Convert the sentences into sequences using Tokenizer.
4. Apply padding to make all sequences the same length.
5. Split the data into input (X) and output (y) using binary labels.
6. Build the Neural Network with an Embedding layer, a Hidden Dense layer using ReLU, a
   second Hidden Dense layer using Tanh, and an Output Dense layer using Sigmoid.
7. Compile the model with the Adam optimizer and binary_crossentropy loss.
8. Train the model using the prepared dataset.
9. Enter a new sentence to predict its sentiment.
10. Display the predicted sentiment (Positive/Negative) as the output.

## CODE

```python
# Import Libraries
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Read Dataset from CSV File (200 rows, columns: sentence, label)
df = pd.read_csv("lab3_sentiment_dataset.csv")

sentences = df["sentence"].tolist()
labels = df["label"].tolist()   # 1 = Positive, 0 = Negative

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(sentences)
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding="post")
y = np.array(labels)

# Build Model: Embedding -> ReLU Hidden -> Tanh Hidden -> Sigmoid Output
model = Sequential()
model.add(Embedding(total_words, 16, input_length=max_length))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation="relu"))   # ReLU: removes negative values, fast learning
model.add(Dense(8, activation="tanh"))    # Tanh: outputs zero-centred values (-1 to 1)
model.add(Dense(1, activation="sigmoid")) # Sigmoid: squashes output to 0-1 probability

# Compile Model
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Train Model
model.fit(X, y, epochs=200, verbose=0)

# Predict Sentiment
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding="post")
    prediction = model.predict(padded, verbose=0)[0][0]
    return "Positive" if prediction >= 0.5 else "Negative"

sample = input("Enter a sentence: ")
print("Predicted Sentiment:", predict_sentiment(sample))
```

**Requirements:** Python 3.11, `tensorflow`, `numpy`, `pandas`

**Dataset file required:** `lab3_sentiment_dataset.csv` (200 rows, columns: `sentence`, `label`)
must be placed in the same folder as the script.

## OUTPUT

```
Enter a sentence: the food and service were absolutely wonderful
Predicted Sentiment: Positive
```

## CONCLUSION

A sentiment analysis model was successfully developed using a Neural Network combining
multiple activation functions. ReLU in the first hidden layer allowed fast, sparse learning by
zeroing out negative activations; Tanh in the second hidden layer produced zero-centred
outputs that helped balance the signal passed forward; and Sigmoid in the output layer
compressed the final value into a probability between 0 and 1, allowing a clear Positive/Negative
decision. This experiment demonstrated how the choice of activation function shapes the
behaviour and output of each layer in a Neural Network.

## QUESTIONS ON THE PROGRAM

1. What is the purpose of sentiment analysis in this program?
2. Why do we label sentences as 1 (Positive) or 0 (Negative)?
3. What is the role of the Tokenizer and pad_sequences here?
4. What does the Embedding layer learn during training?
5. Why do we use GlobalAveragePooling1D instead of Flatten in this model?
6. What is the ReLU activation function and why is it used in the first hidden layer?
7. What are the advantages and limitations of ReLU?
8. What is the Tanh activation function and how does its output range differ from ReLU?
9. Why might Tanh be useful in a hidden layer after ReLU?
10. What is the Sigmoid activation function and why is it used in the output layer?
11. Why is sigmoid preferred over softmax for this binary classification task?
12. What is binary_crossentropy and why is it suitable here?
13. How does the RBF (Radial Basis Function) activation differ conceptually from ReLU, Sigmoid, and Tanh?
14. Why is the Adam optimizer used to train this model?
15. How is the final sentiment decision made from the sigmoid output?
16. What would happen if all activation functions in the network were linear?
17. Why is a small dataset like this one likely to overfit?
18. How does model.predict() differ from model.fit() in this program?
19. What could cause the model to misclassify a sarcastic sentence?
20. How could using an LSTM layer instead of GlobalAveragePooling1D change the model's performance?
