import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the trained weights
model.load_weights('negation_detection.h5')

# Define a new sentence to predict
new_sentence = "I don't like this product"

# Tokenize the new sentence
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts([new_sentence])
sequences = tokenizer.texts_to_sequences([new_sentence])
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')

# Make a prediction on the new sentence
prediction = model.predict(padded_sequences)

# Print the prediction
print(prediction)
