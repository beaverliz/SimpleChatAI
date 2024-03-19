import requests
import json

url = 'http://127.0.0.1:5000/getResponseFromTheModel'
data = {'requestMessage': 'IMS 3.0  как зайти в систему'}

headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)

# Открываем файл для записи
with open(r'SimpleChatAI\\data\\response_output.txt', 'w',encoding='utf-8') as file:
    # Записываем ответ в файл
    file.write(response.text)