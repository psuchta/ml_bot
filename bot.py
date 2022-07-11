import json
import string
import random 

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout

nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

def get_chat_definitions(file_name):
  with open(file_name, 'r') as f:
    return json.load(f)

def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
  return tokens

def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels): 
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list

def get_response(intents_list, intents_json): 
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents: 
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result

url_questions = ["How can I $", "I can't find $ page", "Where is $", "$ doesn't work", "I have problem with $", "I can't do $", "I have to $", "Help me with $"]

account_questions = get_chat_definitions('account_questions.json')
packages_questions = get_chat_definitions('olx_packages.json')

data = account_questions
data['intents'].extend(packages_questions['intents'])

lemmatizer = WordNetLemmatizer()
words = []
classes = []
doc_X = []
doc_y = []
doc_url = []

for intent in data["intents"]:
  if intent["is_url_question"]:
    for word in intent["fill_with_words"]:
      for question in url_questions:
        full_question = question.replace('$', word)
        doc_X.append(full_question)
        doc_y.append(intent["tag"])
        tokens = nltk.word_tokenize(full_question)
        words.extend(tokens)

        if intent["tag"] not in classes:
          classes.append(intent["tag"])
    
  for pattern in intent["patterns"]:
    tokens = nltk.word_tokenize(pattern)
    words.extend(tokens)
    doc_X.append(pattern)
    doc_y.append(intent["tag"])

    if intent["tag"] not in classes:
      classes.append(intent["tag"])


words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))


training = []
out_empty = [0] * len(classes)
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # to
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)

while True:
    message = input("")
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(f'Bot Response - {result}')

