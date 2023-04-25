# First, we need to prepare our data. 
# Let's assume we have a labeled dataset of sentences, where each sentence is labeled as containing negation or not.
# We'll use a binary classifier to predict the presence of negation in a sentence.
# We can start by importing the necessary libraries:

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#  load our dataset and split it into training and testing sets

sentences = ["The movie was not bad", "The food was great"]
labels = [1, 0]  # 1 indicates negation, 0 indicates no negation

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

padded_sequences = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')

train_data = padded_sequences[:1]
train_labels = labels[:1]

test_data = padded_sequences[1:]
test_labels = labels[1:]


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_data, test_labels)

print('Test accuracy:', test_acc)
