A Python-based AI chatbot that is implemented with Flask REST API.

Library requirements:
-Flask
-TensorFlow

Various files that are required:
-style.css
-css.css
-index.html
-train.py
-app.py


In order to run the codes,

Run the train.py script to start training the model. A file called chatbot_model.h5 will be created. The model that the Flask REST API will use to provide responses without requiring retraining is contained in this file. After executing train.py, run app.py to start and initialize the chatbot. You can add more terms and vocabulary to the bot's understanding by editing the intents.json file, adding your own customized words, and then retraining the model.
