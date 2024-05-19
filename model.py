import gensim
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.src.utils import pad_sequences
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.layers import Dense, Embedding, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras import Sequential
import keras
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors


# Model init
def build_model(X_train, y_train, X_test, y_test, vocabulary_length, weights):
    model = Sequential([
        Embedding(vocabulary_length, 300, weights=[weights], trainable=False),
        SpatialDropout1D(0.5),
        Conv1D(64, 5, activation='relu', padding='same'),
        MaxPooling1D(),
        Dropout(0.5),
        Conv1D(128, 5, activation='relu', padding='same'),
        MaxPooling1D(),
        Dropout(0.3),
        Conv1D(256, 5, activation='relu', padding='same'),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=keras.optimizers.Adam(learning_rate=0.001))
    model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))
    return model


def preprocess_data(filepath, tokenizer):
    data = pd.read_csv(filepath, sep="\t", on_bad_lines='skip')
    data.columns = ["text"]
    data_clean = data['text'].apply(lambda x: gensim.utils.simple_preprocess(x))
    text = tokenizer.texts_to_sequences(data_clean)
    return pad_sequences(text, 75)


# Train data load and preprocessing
train_data = pd.read_csv("train/train.tsv", sep="\t", on_bad_lines='skip')
train_data.columns = ["label", "text"]
train_data_cleaned = train_data['text'].apply(lambda x: gensim.utils.simple_preprocess(x))

tokenizer = Tokenizer(10000)
tokenizer.fit_on_texts(train_data_cleaned)
text = tokenizer.texts_to_sequences(train_data_cleaned)
text = pad_sequences(text, 75)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(np.array(text), train_data['label'], test_size=0.2)

# Initializing w2v model
vocabulary_length = len(tokenizer.word_index)
w2v_model = gensim.models.Word2Vec(train_data_cleaned,
                                   vector_size=300)
weights = np.array(
    [w2v_model.wv[word] if word in w2v_model.wv else np.zeros(300) for word, i in tokenizer.word_index.items()])

model = build_model(X_train, y_train, X_test, y_test, vocabulary_length, weights)

model.save('model.h5')

# Loading model and target data.
loaded_model = tf.keras.models.load_model('model.h5')
dev_data = preprocess_data('./dev-0/in.tsv', tokenizer)
test_data = preprocess_data('./test-A/in.tsv', tokenizer)

# Running predictions
y_dev_pred = loaded_model.predict(dev_data)
y_test_pred = loaded_model.predict(test_data)

dev_predictions = np.where(y_dev_pred >= 0.5, 1, 0)
test_predictions = np.where(y_test_pred >= 0.5, 1, 0)

# Saving results
dev_output = pd.DataFrame(dev_predictions)
dev_output.to_csv('./dev-0/out.tsv', sep='\t', index=False)

test_output = pd.DataFrame(test_predictions)
test_output.to_csv('./test-A/out.tsv', sep='\t', index=False)
