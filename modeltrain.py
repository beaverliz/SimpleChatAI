import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,LSTM,Embedding
from keras.optimizers import SGD, Adam
import random
from keras.optimizers import Adam
nltk.data.path.append('SimpleChatAI\\data\\nltk_data')  # путь до папки с данными

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('SimpleChatAI\\data\\intents.json',encoding='utf-8').read()
intents = json.loads(data_file)

lemmatizer = WordNetLemmatizer()

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        tag = str(intent['tag'])  # преобразование в строку
        if tag not in classes:
            classes.append(tag)

# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
#print(len(documents), "documents")
# classes = intents
#print(len(classes), "classes", classes)
# words = all words, vocabulary
#print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('SimpleChatAI/data/words.pkl', 'wb'))
pickle.dump(classes, open('SimpleChatAI/data/classes.pkl', 'wb'))


training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)


    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.asarray(training, dtype="object")
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Создание модели
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Компиляция модели
sgd = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Обучение модели
hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Сохранение модели
model.save('SimpleChatAI/data/hackatonnew_model.keras', hist)

#print("Модель создана")
