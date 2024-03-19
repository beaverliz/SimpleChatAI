
from flask import Flask, request, jsonify
import json
from datetime import datetime, timezone
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import random
import numpy as np
import pickle
from keras.models import load_model

app = Flask(__name__)

nltk.data.path.append('./data/nltk_data')  # путь до папки с данными

# Загрузка данных и модели
words = pickle.load(open('./data/words.pkl', 'rb'))
classes = pickle.load(open('./data/classes.pkl', 'rb'))
model = load_model('./data/hackaton_model.keras')
intents = json.loads(open('./data/intents.json', encoding='utf-8').read())


def clean_up_sentence(sentence):
    # Токенизация и лемматизация ввода пользователя
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    # Создание мешка слов для ввода пользователя
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence):
    # Получение вероятностного распределения для каждого класса
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Сортировка результатов по вероятности
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result



@app.route('/getResponseFromTheModel', methods=['POST'])
def get_response_from_model():
    try:
        data = request.get_json()
        request_message = data.get("requestMessage", "")

        response_message = ""
            # Взаимодействие с моделью
        try:
                # Предсказание класса для ввода пользователя
                predicted_intents = predict_class(request_message)
                # Получение ответа на основе предсказанного класса
                response_message = get_response(predicted_intents, intents)
        except Exception as e:
                # Если произошла ошибка при взаимодействии с моделью, возвращаем сообщение об ошибке
                response_message = "Изивни, галактический друг. Пока на твой вопрос я не могу дать ответ, меня еще обучают. Но ты можешь попробовать написать письмо на межзвездную почту! "

        return jsonify({
            "responseMessage": response_message,
            "responseTime": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500          

if __name__ == '__main__':
    app.run(debug=True)
