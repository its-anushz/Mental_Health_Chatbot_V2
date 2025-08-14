import json
import random
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("punkt")
nltk.download("wordnet")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
with open("ChatbotWebsite/static/data/intents.json") as file:
    intents = json.load(file)

# We will create these files, so we wrap in a try/except block for subsequent runs
try:
    # Load tokenized data
    with open("data.pickle", "rb") as f:
        words, classes, tokenizer, max_len = pickle.load(f)
    model = load_model("chatbot-model.h5")
except:
    words = []
    classes = []
    documents = []
    ignore_words = ["?", "!", ".", ","]

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))
    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique lemmatized words", words)

    training_sentences = []
    training_labels = []
    labels_map = {label: i for i, label in enumerate(classes)}

    for doc in documents:
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
        training_sentences.append(" ".join(pattern_words))
        training_labels.append(labels_map[doc[1]])
        
    tokenizer = Tokenizer(num_words=len(words), oov_token="<OOV>")
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    
    max_len = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    training_labels_final = np.array(training_labels)

    # =================================================================
    # NEW: GloVe Embedding Integration
    # =================================================================
    
    # 1. Load the GloVe vectors from the file
    embedding_dim = 100  # Must match the GloVe file dimension (e.g., 100d)
    embeddings_index = {}
    with open('glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"Found {len(embeddings_index)} word vectors in GloVe.")

    # 2. Create an embedding matrix for our vocabulary
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # =================================================================
    # MODIFIED: BiLSTM Model Architecture to use GloVe
    # =================================================================
    
    model = Sequential()
    # Use the GloVe embedding matrix as weights in the Embedding layer
    # We set trainable=False to prevent the pre-trained weights from being updated during training.
    model.add(Embedding(vocab_size, 
                        embedding_dim, 
                        weights=[embedding_matrix], 
                        input_length=max_len, 
                        trainable=False))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(classes), activation='softmax'))
    
    adam = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    
    # Training the model
    history = model.fit(padded_sequences, training_labels_final, epochs=100, batch_size=10, verbose=1)
    model.save("chatbot-model.h5")

    # Save tokenizer and other necessary data
    with open("data.pickle", "wb") as f:
        pickle.dump((words, classes, tokenizer, max_len), f)
        
    print("Done. Model with GloVe embeddings created and saved.")

# --- The rest of the functions (predict_class, get_response) remain the same ---

def predict_class(message, ERROR_THRESHOLD=0.25):
    message_words = [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(message)]
    message_sequence = tokenizer.texts_to_sequences([" ".join(message_words)])
    padded_message = pad_sequences(message_sequence, maxlen=max_len, padding='post')
    res = model.predict(padded_message)[0]
    results = []
    for i, r in enumerate(res):
        if r > ERROR_THRESHOLD:
            results.append((classes[i], r))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def get_response(message, id="000"):
    results = predict_class(message)
    if results:
        tag = results[0][0]
        for intent in intents["intents"]:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                return str(response)
    return "I'm sorry, I don't quite understand. Could you please rephrase that?"

if __name__ == "__main__":
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_message = input("You: ")
        if user_message.lower() == 'quit':
            break
        
        bot_response = get_response(user_message)
        print(f"Bot: {bot_response}")