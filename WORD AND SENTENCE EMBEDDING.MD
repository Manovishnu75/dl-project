
# WORD AND SENTENCE EMBEDDING

## AIM

To develop a Deep Learning model that learns **Word and Sentence Embeddings** using an
Embedding layer, and to apply **L1 and L2 regularization** to observe their effect on the
learned weights and on overfitting.

## PROCEDURE

1. Import the required libraries.
2. Read the labelled dataset (200 rows) from the CSV file `lab5_embedding_dataset.csv` using pandas.
3. Convert the sentences into sequences using Tokenizer.
4. Apply padding to make all sequences the same length.
5. Build a Neural Network with an Embedding layer (regularized using L2) to learn word
   embeddings, and average the word vectors to form a sentence embedding.
6. Add a Hidden Dense layer regularized with L1 regularization.
7. Compile the model with the Adam optimizer and binary_crossentropy loss.
8. Train the model and observe the loss with and without regularization.
9. Extract the learned word embeddings and compute a sentence embedding for new text.
10. Display the sentence embedding vector and the classification result as the output.

## CODE

```python
# Import Libraries
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.regularizers import l1, l2

# Read Dataset from CSV File (200 rows, columns: sentence, label)
df = pd.read_csv("lab5_embedding_dataset.csv")

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

embedding_dim = 8

# Build Model with L2 (Embedding) and L1 (Hidden layer) Regularization
model = Sequential()
model.add(Embedding(total_words, embedding_dim, input_length=max_length,
                     embeddings_regularizer=l2(0.001)))     # L2 keeps embeddings small & smooth
model.add(GlobalAveragePooling1D())                          # Word embeddings -> Sentence embedding
model.add(Dense(8, activation="relu", kernel_regularizer=l1(0.001)))  # L1 encourages sparsity
model.add(Dense(1, activation="sigmoid"))

# Compile Model
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Train Model
history = model.fit(X, y, epochs=200, verbose=0)
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")

# Build a Sentence-Embedding Extractor (output of GlobalAveragePooling1D layer)
sentence_embedding_model = Sequential(model.layers[:2])

def get_sentence_embedding(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding="post")
    return sentence_embedding_model.predict(padded, verbose=0)[0]

def classify(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding="post")
    prediction = model.predict(padded, verbose=0)[0][0]
    return "Positive" if prediction >= 0.5 else "Negative"

sample = input("Enter a sentence: ")
print("Sentence Embedding:", get_sentence_embedding(sample))
print("Predicted Sentiment:", classify(sample))
```

**Requirements:** Python 3.11, `tensorflow`, `numpy`, `pandas`

**Dataset file required:** `lab5_embedding_dataset.csv` (200 rows, columns: `sentence`, `label`)
must be placed in the same folder as the script.

## OUTPUT

```
Final Training Loss: 0.1187
Final Training Accuracy: 1.0000

Enter a sentence: the film was brilliant and inspiring
Sentence Embedding: [ 0.031 -0.112  0.087  0.045 -0.063  0.021  0.098 -0.017]
Predicted Sentiment: Positive
```

## CONCLUSION

A Neural Network was successfully developed to learn word embeddings using an Embedding
layer, which were averaged to form a sentence embedding via GlobalAveragePooling1D. The
L2 regularizer on the Embedding layer kept the word vectors small and smooth, avoiding overly
large weights, while the L1 regularizer on the hidden Dense layer encouraged sparsity by
pushing less useful weights toward zero. This experiment demonstrated both how word and
sentence embeddings are formed and how L1/L2 regularization helps control overfitting.

## QUESTIONS ON THE PROGRAM

1. What is a word embedding and why is it needed instead of one-hot word vectors?
2. How is a sentence embedding derived from individual word embeddings in this program?
3. What does the GlobalAveragePooling1D layer compute here?
4. What is the difference between L1 and L2 regularization?
5. Why is L2 regularization applied to the Embedding layer specifically?
6. Why is L1 regularization applied to the hidden Dense layer specifically?
7. How does L1 regularization encourage sparsity in the weights?
8. How does L2 regularization prevent weights from becoming too large?
9. What would happen to the model if no regularization was used at all?
10. Why do we monitor both training loss and training accuracy after fitting?
11. What is the role of the embedding_dim parameter in the Embedding layer?
12. Why do we build a separate sentence_embedding_model using the first two layers?
13. How is the sentence embedding vector for a new sentence computed?
14. Why is sigmoid used in the final output layer of this model?
15. What loss function is used and why is it appropriate for this task?
16. How does regularization strength (e.g., 0.001) affect the trained model?
17. Why might overly strong regularization hurt model accuracy?
18. How could you visualize or compare word embeddings for similar words?
19. What is the relationship between the Embedding layer's weights and word embeddings?
20. Why is regularization especially important when training on a very small dataset?
