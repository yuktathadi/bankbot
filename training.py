#TRAINING.PY

# Importing the libraries necessary
from keras.callbacks import EarlyStopping
import random
from keras.models import load_model
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Dropout
from tensorflow.keras import layers, models
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import nltk
import pickle
import json
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#nltk.download('omw-1.4')
#nltk.download("punkt")
#nltk.download("wordnet")


# Initialization of the intents/phrases using the intentsfile.json file
wordcount = []
classes = []
docs = []
ignore_wordcount = ["?", "!"]
intents_file = open("intentsfile.json").read()
intents = json.loads(intents_file)

# For the words
for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        # We shall consider one word at a time and tokenize it
        wrd = nltk.word_tokenize(pattern)
        wordcount.extend(wrd)
        # Now, appending the file to the code
        docs.append((wrd, intent["tag"]))

        # Adding classeses to the list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Performing lemmatization
wordcount = [lemmatizer.lemmatize(wrd.lower()) for wrd in wordcount if wrd not in ignore_wordcount]
wordcount = sorted(list(set(wordcount)))

classes = sorted(list(set(classes)))

print(len(docs), "Documents")

print(len(classes), "Classes", classes)

print(len(wordcount), "Lemmatized unique word", wordcount)


pickle.dump(wordcount, open("wordcount.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Now we train the initializer and training the data
training_process = []

empty_op = [0] * len(classes)
for doc in docs:

    # Initializing the wordcount bag
    wc_bag = []
 
   # listing the tokenized wordcount for the bag
    pattern_wordcount = doc[0]
   
 # Lemmatizing each word i.e., creating a base word to represent related wordcount
    pattern_wordcount = [lemmatizer.lemmatize(word.lower()) for word in pattern_wordcount]
    
# Creating our bag of wordcount array with 1 if the word has been matched in the current pattern
    for wrd in wordcount:
        wc_bag.extend([1]) if wrd in pattern_wordcount else wc_bag.extend([0])

    # 0, for the output for each tag and '1' for the current tag for each pattern
    row_op = list(empty_op)
    row_op[classes.index(doc[1])] = 1
    training_process.append([wc_bag, row_op])

# Shuffling the features and processing into numpy array
random.shuffle(training_process)
training_process=np.array(training_process, dtype=object)
#training_process = np.array(training_process)

# Creating training and testing lists, where X is for patterns and Y is for intents
trainX = list(training_process[:, 0])
trainY = list(training_process[:, 1])
print("The data for training process has been created")

# Creating a model with three layers- 128 neurons in the first, 64 neurons in the second, trainY[0] number of neurons in the third output layer

model = Sequential()
model.add(Dense(128, input_shape=(len(trainX[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(trainY[0]), activation="softmax"))
model.summary()

# Compiling the model
# The Stochastic gradient descent along with Nesterov accelerated gradient will result in good outputs for our model
#stocgraddesc = tf.keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
stocgraddesc = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=stocgraddesc, metrics=["accuracy"])

# To determine the ideal number of training epochs while mitigating the risks of underfitting or overfitting, we employ an early stopping callback in Keras. This can be based on monitoring either loss or accuracy. When monitoring the loss, the training process stops if an increase in the loss values is detected. Conversely, when monitoring accuracy, the training halts upon observing a decrease in accuracy values.
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Now, fitting and then saving the model
model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1, callbacks=[early_stopping])
#savemodel = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save("chatbot_modell.h5")
print("The model has been created")

