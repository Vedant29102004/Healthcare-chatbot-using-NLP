import nltk
import numpy as np
import random
import pickle
import json
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
nltk.download('punkt_tab')

import nltk
import numpy as np
import random
import pickle
import json
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
nltk.download('punkt_tab')

intents = json.loads(open("./content/intents.json").read())
words = pickle.load(open("content\words.pkl", "rb"))
classes = pickle.load(open("content\classes.pkl", "rb"))
model = load_model("content\chatbot_model.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list):
    # Check if intents_list is empty
    if intents_list:  
        tag = intents_list[0]['intent']
        for i in intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    else:
        # Return a default response if no intent is recognized
        return "I'm sorry, I didn't understand that." 

def start_chat():
    print("MediBot is ready to assist.")
    while True:
        inp = input("You: ")
        if inp.lower() in ["quit", "exit", "bye"]:
            print("MediBot: Take care! 👋")
            break
        intents_list = predict_class(inp)
        response = get_response(intents_list)
        print("MediBot:", response)

if __name__ == "__main__":
    start_chat()
