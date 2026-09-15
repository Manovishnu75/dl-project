
# NEURAL NETWORK OVERVIEW – DIALOGUE SYSTEM

## AIM

To develop a Deep Learning model using a simple feedforward **Neural Network (NN)** to build a
basic dialogue system (chatbot) that classifies user input into an intent and returns an
appropriate response, thereby understanding the basic architecture and working of a Neural
Network.

## PROCEDURE

1. Import the required libraries.
2. Read the intent dataset (200 rows) from the CSV file `lab1_dialogue_dataset.csv` using pandas.
3. Convert the text patterns into sequences using Tokenizer.
4. Apply padding to make all sequences the same length.
5. Encode the intent labels using LabelEncoder and one-hot encoding.
6. Build a feedforward Neural Network with an Embedding (Input) layer, a Hidden layer, and an Output layer.
7. Compile the model with the Adam optimizer and categorical_crossentropy loss.
8. Train the model using the prepared dataset.
9. Enter a user query and predict its intent using the trained NN.
10. Display the matching response as the chatbot's output.

## CODE

```python
# Import Libraries
import numpy as np
import pandas as pd
import random

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Read Dataset from CSV File (200 rows, columns: intent, pattern)
df = pd.read_csv("lab1_dialogue_dataset.csv")

patterns = df["pattern"].tolist()
labels = df["intent"].tolist()

# Canned Responses for Each Intent
responses = {
    "greeting": ["Hello! How can I help you today?"],
    "goodbye": ["Goodbye! Have a great day."],
    "thanks": ["You're welcome!"],
    "complaint": ["I'm sorry to hear that. Let me help resolve this."],
    "request_help": ["Sure, I can help you with that."],
    "order_status": ["Let me check that for you right away."],
    "book_appointment": ["Sure, let's get that scheduled for you."],
    "weather_query": ["Let me check the forecast for you."]
}

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
total_words = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(patterns)
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding="post")

# Encode Labels
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(labels)
y = to_categorical(y_int)
num_classes = y.shape[1]

# Build Neural Network: Input -> Hidden -> Output
model = Sequential()
model.add(Embedding(total_words, 8, input_length=max_length))   # Input layer
model.add(GlobalAveragePooling1D())                              # Flattens input for hidden layer
model.add(Dense(16, activation="relu"))                          # Hidden layer
model.add(Dense(num_classes, activation="softmax"))              # Output layer

# Compile Model
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Train Model
model.fit(X, y, epochs=200, verbose=0)

# Dialogue Response Function
def get_response(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding="post")
    prediction = model.predict(padded, verbose=0)
    intent = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return random.choice(responses[intent])

# Test the Dialogue System
user_input = input("You: ")
print("Bot:", get_response(user_input))
```

**Requirements:** Python 3.11, `tensorflow`, `scikit-learn`, `numpy`, `pandas`
(`pip install tensorflow scikit-learn numpy pandas`)

**Dataset file required:** `lab1_dialogue_dataset.csv` (200 rows, columns: `intent`, `pattern`) must
be placed in the same folder as the script.

## OUTPUT

```
You: hey, how are you doing?
Bot: Hello! How can I help you today?
```

## CONCLUSION

A basic dialogue system was successfully developed using a feedforward Neural Network. The
Embedding layer acted as the input layer converting words into dense vectors, the Dense layer
with ReLU activation served as the hidden layer that learned intermediate patterns, and the
softmax Dense layer acted as the output layer producing the predicted intent. This experiment
demonstrated the fundamental Input–Hidden–Output architecture of a Neural Network in a
real-world dialogue system.

## QUESTIONS ON THE PROGRAM

1. What is a Neural Network and what are its three main types of layers?
2. Why do we group user patterns by intent in this program?
3. What is the role of the Tokenizer in converting text to numbers?
4. Why is padding applied to the tokenized sequences?
5. What does the Embedding layer do in this model?
6. Why do we use GlobalAveragePooling1D after the Embedding layer?
7. What is the difference between the Input layer and the Hidden layer here?
8. Why is ReLU activation used in the hidden Dense layer?
9. Why is softmax activation used in the output layer?
10. Why do we one-hot encode the intent labels using to_categorical?
11. What does LabelEncoder do in this program?
12. Why is categorical_crossentropy chosen as the loss function?
13. What is the role of the Adam optimizer during training?
14. What happens internally when model.fit() is called for 200 epochs?
15. How is the user's input converted into a form the model understands?
16. How does the model choose which intent to predict?
17. Why do we use np.argmax on the model's prediction?
18. What is the purpose of the random.choice() call in get_response()?
19. What would happen if the training dataset had very few examples per intent?
20. How could this basic dialogue system be extended to handle more intents?
