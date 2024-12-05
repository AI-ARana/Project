# Gated Recurrent Unit (GRU) with TensorFlow and Keras

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Sample text data
text = (
    "Anurag Rana, an Associate Professor and Scientist delves into AI and AGI, "
    "bringing his expertise to interdisciplinary realms with active involvement."
)

# Tokenizing the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1  # Add 1 for padding token

# Reverse word_index for decoding predictions
index_word = {index: word for word, index in tokenizer.word_index.items()}

# Generate input sequences using n-grams
input_sequences = []
words = text.split()
for i in range(1, len(words)):
    n_gram_sequence = words[:i + 1]
    token_list = tokenizer.texts_to_sequences([' '.join(n_gram_sequence)])[0]
    input_sequences.append(token_list)

# Pad sequences
max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and labels
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Define the model
model = Sequential([
    Embedding(input_dim=total_words, output_dim=10),
    GRU(150, return_sequences=False),
    Dense(total_words, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]
    return index_word.get(predicted_index, "<unknown>")

# Example usage
seed_text = "Anurag Rana, an Associate"
next_word = predict_next_word(model, tokenizer, seed_text, max_sequence_len)
print(f"Predicted next word: {next_word}")
