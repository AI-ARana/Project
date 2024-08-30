# Gated Recurrent Unit

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Sample text data
text = "Anurag Rana, an Associate Professor and Scientist delves into AI and AGI, bringing his expertise to interdisciplinary realms with active involvement."

# Tokenizing the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Convert text to sequence of tokens
input_sequences = []
for i in range(1, len(text.split())):
    n_gram_sequence = text.split()[:i+1]
    token_list = tokenizer.texts_to_sequences([' '.join(n_gram_sequence)])[0]
    input_sequences.append(token_list)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and label
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Define the model
model = Sequential([
    Embedding(total_words, 10, input_length=max_sequence_len-1),
    GRU(150),
    Dense(total_words, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]
    return tokenizer.index_word[predicted_index]

# Example usage
seed_text = "Anurag Rana, an Associate"
next_word = predict_next_word(model, tokenizer, seed_text, max_sequence_len)
print(f"Predicted next word: {next_word}")
