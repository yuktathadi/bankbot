#MAINAPPL.PY


# libraries
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatize = WordNetLemmatizer()


# Initializing the conversation chat
model = load_model("chatbot_modell.h5")
intents = json.loads(open("intentsfile.json").read())
word = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)
# run_with_ngrok(app) 

@app.route("/")
def home():
    return render_template("indexfile.html")


@app.route("/get", methods=["POST"])
def bankbot_response():
    msg = request.form["msg"]
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        result1 = responseobtained(ints, intents)
        result =result1.replace("{n}",name)
    elif msg.startswith('Hello, this is Bankbot'):
        name = msg[14:]
        ints = predict_class(msg, model)
        result1 = responseobtained(ints, intents)
        result =result1.replace("{n}",name)
    else:
        ints = predict_class(msg, model)
        result = responseobtained(ints, intents)
    return result


# Functionality of bankbot
def clean_sentence(sentence):
    stnce_tknz = nltk.word_tokenize(sentence)
    stnce_tknz = [lemmatize.lemmatize(word.lower()) for word in stnce_tknz]
    return stnce_tknz


# Returning the bag of words array. Returns 0 or 1 for each word in the bag that does exist in the sentences

def bagowords(sentence, word, show_details=True):
  
  # tokenize the pattern
    stnce_tknz = clean_sentence(sentence)
    # bag of word - matrix of N word, vocabulary matrix
    bag = [0] * len(word)
    for s in stnce_tknz:
        for i, wrd in enumerate(word):
            if wrd == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bagowords(sentence, word, show_details=False)
    result = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results_obtained = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results_obtained.sort(key=lambda x: x[1], reverse=True)
    return_list_bot = []
    for r in results_obtained:
        return_list_bot.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list_bot


def responseobtained(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


if __name__ == "__main__":
    app.run(port=5000)
