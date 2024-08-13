text = "Generative AI is primarily designed to create or generate new content. This content can be text, images, music, videos, or even code."
import sys
import io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create character mappings
chars = sorted(list(set(text)))
char_to_index = {ch: i for i, ch in enumerate(chars)}
index_to_char = {i: ch for i, ch in enumerate(chars)}

# Convert text to indices
text_as_int = np.array([char_to_index[ch] for ch in text])

# Prepare sequences
seq_length = 10
X = []
y = []
for i in range(len(text_as_int) - seq_length):
    X.append(text_as_int[i:i + seq_length])
    y.append(text_as_int[i + seq_length])

X = np.array(X)
y = to_categorical(y, num_classes=len(chars))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding

model = Sequential()
model.add(Embedding(input_dim=len(chars), output_dim=10, input_length=seq_length))
model.add(SimpleRNN(50, return_sequences=False))
model.add(Dense(len(chars), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def generate_text(seed_text, length=100):
    # Ensure seed_text is of the same length as the sequence length
    seed_text = seed_text[:seq_length]  # Truncate if longer
    generated_text = seed_text

    for _ in range(length):
        # Convert seed_text to a sequence of integers
        sequence = np.array([char_to_index[ch] for ch in seed_text]).reshape(1, -1)
        
        # Predict the next character
        
        #pred_index = np.argmax(model.predict(sequence), axis=-1)[0]
        #pred_char = index_to_char[pred_index]
        # Generate a new character based on the current sequence
        pred_index = np.argmax(model.predict(sequence), axis=-1)

        # Access the first element if pred_index is an array
        if isinstance(pred_index, np.ndarray):
            pred_index = pred_index[0]
    
        pred_char = index_to_char[pred_index]
        
        # Add the predicted character to the generated text
        generated_text += pred_char
        
        # Update the seed_text to be the last 'seq_length' characters
        seed_text = generated_text[-seq_length:]

    return generated_text

seed_text = "AI"  # Original text
generated_text = generate_text(seed_text, length=100)
print(generated_text)
