import pandas as pd
import json

# Загрузка данных из Excel файла
data_path = './data/Датасет для хакатона.xlsx'
intents_path = './data/intents.json'
df = pd.read_excel(data_path)

# Создание пустого списка для хранения intents
intents = []

# Проход по каждой строке в датасете
for index, row in df.iterrows():
    intent = {
        "tag": [""],
        "patterns": row.iloc[0],
        "responses": row.iloc[1],
    }
    intents.append(intent)

# Создание словаря с ключом "intents" и значением списка intents
data = {"intents": intents}

# Преобразование словаря в JSON и сохранение в файл
with open(intents_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)