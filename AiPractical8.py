import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Prepare a dataset of labeled emails (spam and non-spam)
emails = [
    "Buy cheap watches! Free shipping!",
    "Meeting for lunch today?",
    "Claim your prize! You've won $1,000,000!",
    "Important meeting at 3 PM."
]
labels = [1, 0, 1, 0]

# Step 2: Tokenize and pad the email text data
max_words = 1000  # Maximum number of words to consider
max_len = 50  # Maximum length of each email (in terms of words)

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")  # Out of Vocabulary token
tokenizer.fit_on_texts(emails)
sequences = tokenizer.texts_to_sequences(emails)
X_padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

# Step 3: Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=16, input_length=max_len),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])	
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Define training data and labels as NumPy arrays
training_data = np.array(X_padded)
training_labels = np.array(labels)


# Step 5: Train the model
model.fit(training_data, training_labels, epochs=10)

# Step 6: Test if 'Spam.txt' is spam or not
file_path = "Spam.txt"
try:
    with open(file_path, "r", encoding="utf-8") as file:
        sample_email_text = file.read()
    sequences_sample = tokenizer.texts_to_sequences([sample_email_text])
    sample_email_padded = pad_sequences(sequences_sample, maxlen=50, padding="post", truncating="post")
    prediction = model.predict(sample_email_padded)
    if prediction > 0.5:
        print(f"Sample Email ('{file_path}'): SPAM")
    else:
        print(f"Sample Email ('{file_path}'): NOT SPAM")
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

