# Sample dataset with more conversations
training_data = [
    {"intent": "greeting", "text": "Hello"},
    {"intent": "greeting", "text": "Hi"},
    {"intent": "greeting", "text": "Hey"},
    {"intent": "greeting", "text": "Good morning"},
    {"intent": "greeting", "text": "Good evening"},
    {"intent": "greeting", "text": "Howdy"},
    {"intent": "goodbye", "text": "Bye"},
    {"intent": "goodbye", "text": "Goodbye"},
    {"intent": "goodbye", "text": "See you later"},
    {"intent": "goodbye", "text": "See you"},
    {"intent": "goodbye", "text": "Take care"},
    {"intent": "thanks", "text": "Thank you"},
    {"intent": "thanks", "text": "Thanks"},
    {"intent": "thanks", "text": "I appreciate it"},
    {"intent": "thanks", "text": "Much obliged"},
    {"intent": "age", "text": "How old are you?"},
    {"intent": "age", "text": "What's your age?"},
    {"intent": "age", "text": "Tell me your age"},
    {"intent": "name", "text": "What is your name?"},
    {"intent": "name", "text": "What's your name?"},
    {"intent": "name", "text": "Tell me your name"},
    {"intent": "weather", "text": "What's the weather like?"},
    {"intent": "weather", "text": "How's the weather?"},
    {"intent": "weather", "text": "Is it raining?"},
    {"intent": "weather", "text": "Is it sunny?"},
]

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess(text):
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [word for word in tokens if word.isalpha()]
    # Convert to lower case
    tokens = [word.lower() for word in tokens]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# Apply preprocessing to the dataset
for data in training_data:
    data['tokens'] = preprocess(data['text'])
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Prepare data for training
texts = [data['text'] for data in training_data]
intents = [data['intent'] for data in training_data]

# Tokenize texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=10)

# Encode intents
intent_index = {intent: idx for idx, intent in enumerate(set(intents))}
labels = [intent_index[intent] for intent in intents]

# Convert labels to categorical
labels = tf.keras.utils.to_categorical(labels, num_classes=len(intent_index))

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(10,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(intent_index), activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(data, labels, epochs=10, batch_size=1, verbose=1)

# Save model and tokenizer
model.save('chatbot_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    import pickle
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

responses = {
    "greeting": "Hello! How can I help you?",
    "goodbye": "Goodbye! Have a nice day!",
    "thanks": "You're welcome!",
    "age": "I am a chatbot, I don't have an age.",
    "name": "I am your friendly chatbot.",
    "weather": "I don't have access to real-time weather information right now."
}

def chatbot_response(text):
    # Preprocess text
    tokens = preprocess(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=10)

    # Predict intent
    prediction = model.predict(padded_sequence)
    intent = list(intent_index.keys())[list(intent_index.values()).index(prediction.argmax())]

    # Generate response
    return responses[intent]

# Interactive loop
print("Start talking with the bot (type 'quit' to stop)!")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    print("Bot:", chatbot_response(user_input))