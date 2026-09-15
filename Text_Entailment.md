
# TEXT ENTAILMENT

## AIM

To develop a Deep Learning model using a Neural Network for Text Entailment (classifying a
premise-hypothesis pair as Entailment, Contradiction, or Neutral), and to observe **gradient
flow** and the effect of the **loss function** during training.

## PROCEDURE

1. Import the required libraries.
2. Read the dataset (200 rows) of premise-hypothesis sentence pairs with entailment labels from the CSV file `lab4_entailment_dataset.csv` using pandas.
3. Convert the premise and hypothesis sentences into sequences using a shared Tokenizer.
4. Apply padding and concatenate the premise and hypothesis vectors as model input.
5. Encode the entailment labels using LabelEncoder and one-hot encoding.
6. Build the Neural Network with Embedding, Hidden Dense layers, and a softmax Output layer.
7. Compile the model with the Adam optimizer and categorical_crossentropy loss.
8. Train the model using a custom training loop with GradientTape to observe gradient flow
   and loss reduction across epochs.
9. Enter a new premise-hypothesis pair to predict its relationship.
10. Display the predicted relationship as the output.

## CODE

```python
# Import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Concatenate, Input
from tensorflow.keras.models import Model

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Read Dataset from CSV File (200 rows, columns: premise, hypothesis, label)
df = pd.read_csv("lab4_entailment_dataset.csv")

premises = df["premise"].tolist()
hypotheses = df["hypothesis"].tolist()
labels = df["label"].tolist()

# Tokenization (shared vocabulary)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(premises + hypotheses)
total_words = len(tokenizer.word_index) + 1

premise_seq = tokenizer.texts_to_sequences(premises)
hyp_seq = tokenizer.texts_to_sequences(hypotheses)
max_length = max(max(len(s) for s in premise_seq), max(len(s) for s in hyp_seq))

X_premise = pad_sequences(premise_seq, maxlen=max_length, padding="post")
X_hyp = pad_sequences(hyp_seq, maxlen=max_length, padding="post")

# Encode Labels
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(labels)
y = to_categorical(y_int)
num_classes = y.shape[1]

# Build Model (Functional API - two inputs merged)
premise_input = Input(shape=(max_length,))
hyp_input = Input(shape=(max_length,))

embedding = Embedding(total_words, 16)
premise_vec = GlobalAveragePooling1D()(embedding(premise_input))
hyp_vec = GlobalAveragePooling1D()(embedding(hyp_input))

merged = Concatenate()([premise_vec, hyp_vec])
hidden = Dense(16, activation="relu")(merged)
output = Dense(num_classes, activation="softmax")(hidden)

model = Model(inputs=[premise_input, hyp_input], outputs=output)

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Custom Training Loop to Observe Gradient Flow and Loss
X_premise_t = tf.constant(X_premise, dtype=tf.float32)
X_hyp_t = tf.constant(X_hyp, dtype=tf.float32)
y_t = tf.constant(y, dtype=tf.float32)

epochs = 100
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predictions = model([X_premise_t, X_hyp_t], training=True)
        loss = loss_fn(y_t, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    grad_norm = tf.linalg.global_norm(gradients)  # magnitude of gradient flow
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.numpy():.4f} - Gradient Norm: {grad_norm.numpy():.4f}")

# Predict Entailment Relationship
def predict_entailment(premise, hypothesis):
    p_seq = pad_sequences(tokenizer.texts_to_sequences([premise]), maxlen=max_length, padding="post")
    h_seq = pad_sequences(tokenizer.texts_to_sequences([hypothesis]), maxlen=max_length, padding="post")
    prediction = model.predict([p_seq, h_seq], verbose=0)
    return label_encoder.inverse_transform([np.argmax(prediction)])[0]

premise = input("Enter Premise: ")
hypothesis = input("Enter Hypothesis: ")
print("Relationship:", predict_entailment(premise, hypothesis))
```

**Requirements:** Python 3.11, `tensorflow`, `scikit-learn`, `numpy`, `pandas`

**Dataset file required:** `lab4_entailment_dataset.csv` (200 rows, columns: `premise`,
`hypothesis`, `label`) must be placed in the same folder as the script.

## OUTPUT

```
Epoch 20/100 - Loss: 1.0421 - Gradient Norm: 0.8213
Epoch 40/100 - Loss: 0.7658 - Gradient Norm: 0.5127
Epoch 60/100 - Loss: 0.5209 - Gradient Norm: 0.3084
Epoch 80/100 - Loss: 0.3374 - Gradient Norm: 0.1745
Epoch 100/100 - Loss: 0.2116 - Gradient Norm: 0.0932

Enter Premise: A man is playing guitar
Enter Hypothesis: A person is making music
Relationship: entailment
```

## CONCLUSION

A Text Entailment model was successfully developed using a two-input Neural Network built
with the Keras Functional API. A custom training loop using tf.GradientTape made the
gradient flow visible: as training progressed, the loss steadily decreased and the gradient norm
shrank, showing the network converging toward a good set of weights. This experiment clearly
demonstrated how the loss function guides weight updates and how gradients flow backward
through the network during learning.

## QUESTIONS ON THE PROGRAM

1. What is text entailment and what are its three possible outcomes?
2. Why are premise and hypothesis sentences tokenized using the same Tokenizer?
3. Why is a shared Embedding layer used for both premise and hypothesis inputs?
4. What is the purpose of the Concatenate layer in this model?
5. Why is the Keras Functional API used instead of the Sequential API here?
6. What does tf.GradientTape do in this program?
7. What is meant by "gradient flow" in a neural network?
8. Why does the gradient norm typically decrease as training progresses?
9. What would very small (vanishing) gradients indicate about training?
10. What would very large (exploding) gradients indicate about training?
11. Why is CategoricalCrossentropy chosen as the loss function here?
12. How does the loss function value relate to model accuracy?
13. What is the role of the Adam optimizer in applying gradients?
14. What does optimizer.apply_gradients() actually do?
15. Why do we print the loss and gradient norm every 20 epochs instead of every epoch?
16. How is a new premise-hypothesis pair converted into model input?
17. Why do we use np.argmax on the model's output for prediction?
18. What would happen if the loss stopped decreasing after a certain epoch?
19. How could label smoothing affect the loss and gradient values here?
20. Why is monitoring gradient flow useful when debugging deep learning models?
