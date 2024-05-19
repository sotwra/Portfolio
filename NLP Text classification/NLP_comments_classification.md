---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.2
  nbformat: 4
  nbformat_minor: 2
  toc:
    base_numbering: 1
    nav_menu: {}
    number_sections: true
    sideBar: true
    skip_h1_title: true
    title_cell: Содержание
    title_sidebar: Contents
    toc_cell: true
    toc_position:
      height: calc(100% - 180px)
      left: 10px
      top: 150px
      width: 184px
    toc_section_display: true
    toc_window_display: true
---

::: {.cell .markdown}
# Анализ текстов на токсичность

Интернет-магазин запускает новый сервис: пользователи могут
редактировать и дополнять описания товаров, как в вики-сообществах.\
Магазину нужен инструмент, который будет искать токсичные комментарии и
отправлять их на модерацию.

**Задача**

-   Создать модель для классификации комментариев на позитивные и
    негативные\
-   Модель должна обеспечивать значение метрики качества *F1* не меньше
    0.75\
-   Имеет значение ресурсоёмкость решения:
    -   сколько требуется времени для подготовки модели к инфренсу?
    -   требуются ли ресурсы GPU для решения задачи?

**Способ решения**

Исследовать несколько сочетаний моделей и способов предобработки текста
и предложить в продакшн оптимальный вариант.\
С заказчиком согласовано исследование следующих моделей:

-   Логистическая регрессия
-   Бустинг CatBoost

А также кодирование текста при помощи:

-   Методики TF-IDF
-   Предобученной нейросети
:::

::: {.cell .markdown}
## Описание данных

Имеется набор размеченных данных по токсичности правок.

Данные находятся в файле `toxic_comments.csv`.\
Столбец *text* в нём содержит текст комментария, а *toxic* --- бинарный
целевой признак (1/0).
:::

::: {.cell .markdown}
## План работы

1.  Загрузить и исследовать данные:
    -   Изучить данные, проверить их на предмет пропусков, аномалий и
        дубликатов
    -   Исследовать данные на дисбаланс классов
2.  Предобработать данные, подготовить отдельные датасеты для
    моделирования:
    -   Очистить данные от спец. символов, цифр, знаков препинания и др.
    -   Повторно проверить на аномалии
    -   Разбить данные на обучающую и тестовую(20%) выборки
    -   Сделать векторизацию тестов по TF-IDF:
        -   лемматизировать текстовые данные
        -   удалить стоп-слова и сгенерировать дополнительные признаки с
            использованием:
            -   униграмм
            -   диграмм
    -   Подобрать подходящую под задачу предобученную нейросеть из
        сообщества Hugging Face и сгенерировать эмбединги текстов
3.  Исследовать модели с использованием обозначенных выше способов
    предобработки:
    -   Обучить модели с подбором гиперпараметров на кросс-валидации:
        -   Логистическая регрессия:
            -   TF-IDF на униграммах
            -   TF-IDF на диграммах
            -   эмбединги от нейросети\
        -   Бустинг CatBoost:
            -   очищенный текст без векторизации *(i)*
            -   эмбединги от нейросети
    -   Сделать сводую таблицу результатов и выбрать оптимальную модель
        и тип предобработки
4.  Тестирование
    -   Сгенерировать \"дамми\" модель для проверки оптимальной модели
        на адекватность
    -   Проверить оптимальную модель и способ предобработки на тестовых
        данных
5.  Составить выводы и рекомендации

*(i) Модель Catboost имеет встроенные средства векторизации текстов -
можно подавать текстовые признаки без кодирования.*
:::

::: {.cell .markdown}
## Подготовка и первичный анализ
:::

::: {.cell .markdown}
### Загрузка библиотек
:::

::: {.cell .code execution_count="45"}
``` python
import pandas as pd
import numpy as np

#Прогресс выполнения и время работы
from tqdm import tqdm
tqdm.pandas()
from time import time

# Работа с текстом

#Нейросети
import torch
from transformers import BertTokenizer, BertModel

#Борьба с регулярными выражениями
import re

# Лемматизация
from nltk.stem import SnowballStemmer

# Стоп-слова для исключения и леммация англ
import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

# Класс для генерации признаков как матрицы TF-IDF для слов
from sklearn.feature_extraction.text import TfidfVectorizer

# Модели 
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from catboost import CatBoostClassifier

# Метрики, разделялки и впомогательные для моделей
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
```
:::

::: {.cell .markdown}
### Настройки и константы
:::

::: {.cell .code execution_count="46"}
``` python
RANDOM_STATE = 12345

torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
```
:::

::: {.cell .markdown}
### Пользовательские функции
:::

::: {.cell .markdown}
Очистка текста

-   Удаляет все спец. символы, знаки препинания и лишние пробелы
-   Приводит данные к единому регистру
:::

::: {.cell .code execution_count="47"}
``` python
def func_clear_text_lower(text):
    t = re.sub(r'[^a-zA-Z]', ' ', text)
    t = " ".join(t.split())
    t = t.lower()
    return t
```
:::

::: {.cell .markdown}
Лемматизация

-   Использует быстрый лемматизатор Snowball для английского языка

*Пробовал лемматизатор из Spacy, работает на порядки медленнее, а
результат по метрике +/- тот же.*
:::

::: {.cell .code execution_count="48"}
``` python
def func_lemmatize_engl(text):
    stemmer = SnowballStemmer(language='english')
    t = list(text.split())
    result = []
    for a in t:
        result.append(stemmer.stem(a))
    result = ' '.join(result)
    return result
```
:::

::: {.cell .markdown}
Генерация эмбедингов с помощью предобученной нейросети

-   Можно задать имя нейросети из комьюнити Hugging Face
-   По умолчанию - BERT для токсичных комментариев, т.к. эта нейросеть
    как раз под задачу данного проекта
-   Работает на CPU или GPU. По умолчанию GPU.
:::

::: {.cell .code execution_count="49"}
``` python
def func_get_embedings(series_to_encode, run_on_gpu=True, hugface_model_name:str='unitary/toxic-bert', batch_limit=100):

    # Сохранение исходного индекса df/series
    series_to_encode_index = series_to_encode.index

    # Загрузка модели и её токенизатора
    tokenizer = BertTokenizer.from_pretrained(hugface_model_name)
    model = BertModel.from_pretrained(hugface_model_name)

    # Токенизация и кодирование
    tokenized_texts = tokenizer.batch_encode_plus(
                                series_to_encode,
                                padding=True,
                                truncation=True,
                                return_tensors='pt',
                                add_special_tokens=True)
    input_ids = tokenized_texts['input_ids']
    attention_mask = tokenized_texts['attention_mask']

    # Найдём максимальный размера батча (делитель без остатка) в заданном лимите 
    while len(series_to_encode) % batch_limit != 0:
        batch_limit -= 1

    batch_size = batch_limit
    embeddings = []

    # Если работаем на GPU, то предварительно переводим модель и все тензоры на GPU
    # Предварительно проверим, доступна ли работа на GPU
    if run_on_gpu:
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} is available. Running on GPU")

            for i in tqdm(range(input_ids.shape[0] // batch_size)):
                model = model.to('cuda')
                batch = torch.LongTensor(input_ids[batch_size*i:batch_size*(i+1)]).to('cuda')
                attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)]).to('cuda')

                # Генерация эмбедингов
                with torch.no_grad():
                    outputs = model(batch, attention_mask=attention_mask_batch)
                    embeddings_batch = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.append(embeddings_batch)
                
        torch.cuda.empty_cache() # Обязательно очистить кэш GPU, иначе возможны сбои kernel           

    else:
        print("Training will run on CPU.")
        for i in tqdm(range(input_ids.shape[0] // batch_size)):
            batch = torch.LongTensor(input_ids[batch_size*i:batch_size*(i+1)])
            attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)])

            # Генерация эмбедингов
            with torch.no_grad():
                outputs = model(batch, attention_mask=attention_mask_batch)
                embeddings_batch = outputs.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(embeddings_batch)

    return pd.DataFrame(np.concatenate(embeddings), index=series_to_encode_index)
```
:::

::: {.cell .markdown}
### Загрузка и знакомство с данными
:::

::: {.cell .code execution_count="50"}
``` python
try:
    df = pd.read_csv('toxic_comments.csv', index_col='Unnamed: 0')
except:
    df = pd.read_csv('/datasets/toxic_comments.csv', index_col='Unnamed: 0')
```
:::

::: {.cell .code execution_count="51"}
``` python
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    Index: 159292 entries, 0 to 159450
    Data columns (total 2 columns):
     #   Column  Non-Null Count   Dtype 
    ---  ------  --------------   ----- 
     0   text    159292 non-null  object
     1   toxic   159292 non-null  int64 
    dtypes: int64(1), object(1)
    memory usage: 3.6+ MB
:::
:::

::: {.cell .code execution_count="52"}
``` python
df.head()
```

::: {.output .execute_result execution_count="52"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>toxic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Explanation\nWhy the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hey man, I'm really not trying to edit war. It's just that this guy is constantly removing relevant information and talking to me through edits instead of my talk page. He seems to care more about the formatting than the actual info.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>"\nMore\nI can't make any real suggestions on improvement - I wondered if the section statistics should be later on, or a subsection of ""types of accidents""  -I think the references may need tidying so that they are all in the exact same format ie date format etc. I can do that later on, if no-one else does first - if you have any preferences for formatting style on references or want to do it yourself please let me know.\n\nThere appears to be a backlog on articles for review so I guess there may be a delay until a reviewer turns up. It's listed in the relevant form eg Wikipedia:Good_article_nominations#Transport  "</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>You, sir, are my hero. Any chance you remember what page that's on?</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="53"}
``` python
print('Количество дубликатов:', df['text'].duplicated().sum())
```

::: {.output .stream .stdout}
    Количество дубликатов: 0
:::
:::

::: {.cell .code execution_count="54"}
``` python
print('Количество пустых текстовых записей:', df.loc[(df['text']=='')|(df['text']==' ')]['text'].count())
```

::: {.output .stream .stdout}
    Количество пустых текстовых записей: 0
:::
:::

::: {.cell .code execution_count="55"}
``` python
print('Средняя длинна текста:', round(df['text'].str.len().mean()))
print('Максимальная длинна текста:', len(df['text'].max()))
```

::: {.output .stream .stdout}
    Средняя длинна текста: 394
    Максимальная длинна текста: 4521
:::
:::

::: {.cell .markdown}
Исследование баланса классов
:::

::: {.cell .code execution_count="56"}
``` python
class_balance = df['toxic'].value_counts(normalize=True).round(3)
print('Доля нетоксичных записей', class_balance[0]*100, '%')
print('Доля токсичных записей', class_balance[1]*100, '%')
```

::: {.output .stream .stdout}
    Доля нетоксичных записей 89.8 %
    Доля токсичных записей 10.2 %
:::
:::

::: {.cell .markdown}
**Выводы первичного анализа данных**

-   Данных (159 тыс. строк) достаточно для тестирования выбранных типов
    моделей и кодирования
-   Пропусков и дубликатов нет
-   Текстовый массив затруднительно обработать простыми аголоритмами,
    т.к. макс. длинна текста 4500, а средняя длинна 394 символа
-   Сильный дисбаланс классов (9 к 1). Потребуется учитывать это при
    использовании Логистической регрессии
-   Текстовые данные \"грязные\", есть спец символы (\\n и др.), разный
    регистр и т.д.
-   Текст не лемматизирован
:::

::: {.cell .markdown}
## Предобработка данных
:::

::: {.cell .markdown}
### Очистка данных
:::

::: {.cell .code execution_count="57"}
``` python
df['text'] = df['text'].apply(func_clear_text_lower)
```
:::

::: {.cell .markdown}
Проверим не образовались ли пустые записи и/или состоящие из пробела
после очистки текста.\
Такое может быть, если текст состоял только из пробелов и/или спец.
символов.

Если таких наблюдений немного, то можно их отбросить.
:::

::: {.cell .code execution_count="58"}
``` python
df['text'].isna().sum()
```

::: {.output .execute_result execution_count="58"}
    0
:::
:::

::: {.cell .code execution_count="59"}
``` python
print('Количество пустых текстовых записей:', df.loc[(df['text']=='')|(df['text']==' ')]['text'].count())
```

::: {.output .stream .stdout}
    Количество пустых текстовых записей: 11
:::
:::

::: {.cell .code execution_count="60"}
``` python
df = df.loc[(df['text']!='')&(df['text']!=' ')]
```
:::

::: {.cell .markdown}
### Разбиение данных на обучающую и тестовую (20%) выборки

-   Т.к. предобработка будет отличаться в зависимости от типа модели,
    заранее разобьём данные
-   Тестовые данные предобработаем под потребности итоговой модели,
    которую будем на них проверять
-   Дисбаланс классов при разбиении не учитываем - **нет гарантии**, что
    на вновь поступающих данных он будет таким же
:::

::: {.cell .code execution_count="61"}
``` python
df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
```
:::

::: {.cell .code execution_count="62"}
``` python
x_train = df_train['text']
y_train = df_train['toxic']

x_test = df_test['text']
y_test = df_test['toxic']
```
:::

::: {.cell .markdown}
### Векторизация по TF-IDF

Лемматизируем текст и подготовим датасеты с генерацией признаков по
TF-IDF:

-   с униграммами
-   с диграммами

Запишем время, необходимое для этих операций, чтобы потом учесть его в
общей скорости обучения соотв. моделей.
:::

::: {.cell .markdown}
Лемматизация
:::

::: {.cell .code execution_count="63"}
``` python
start = time()

x_train_lemmatized = x_train.copy()
x_train_lemmatized = x_train_lemmatized.apply(func_lemmatize_engl)

end = time()

time_lemmatization = end - start

print('Время лемматизации:', time_lemmatization.__round__(1), 'c')
```

::: {.output .stream .stdout}
    Время лемматизации: 34.3 c
:::
:::

::: {.cell .markdown}
#### TF-IDF с униграммами
:::

::: {.cell .code execution_count="64"}
``` python
start = time()

vectorizer_uni = TfidfVectorizer(stop_words=stopwords) 
vectorizer_uni.fit(x_train_lemmatized)
x_train_tfidf_uni = vectorizer_uni.transform(x_train_lemmatized)

end = time()

time_tfidf_uni = end - start

print("Количество полученных признаков:", x_train_tfidf_uni.shape[1])
print("Время генерации:", time_tfidf_uni.__round__(1), 'c')
```

::: {.output .stream .stdout}
    Количество полученных признаков: 115213
    Время генерации: 6.4 c
:::
:::

::: {.cell .markdown}
#### TF-IDF с диграммами
:::

::: {.cell .code execution_count="65"}
``` python
start = time()

vectorizer_di = TfidfVectorizer(stop_words=stopwords, ngram_range=(2,2)) 
vectorizer_di.fit(x_train_lemmatized)
x_train_tfidf_di = vectorizer_di.transform(x_train_lemmatized)

end = time()

time_tfidf_di = end - start

print("Количество полученных признаков:", x_train_tfidf_di.shape[1])
print("Время генерации:", time_tfidf_di.__round__(1), 'с')
```

::: {.output .stream .stdout}
    Количество полученных признаков: 1867778
    Время генерации: 14.1 с
:::
:::

::: {.cell .markdown}
Вот оно как. Наблюдаем взрыв признаков при переходе на диграммы.\
Число признаков увеличилось в 16 раз до 1,8+ млн.\
Для дальнейшей работы моделей это не лучший вариант из-за слишком
долгого времени работы, особенно с подбором гиперпараметров.

Ограничим число признаков при генерации 1 млн. Векторизатор отберёт их
по частоте встречаемости в корпусе.
:::

::: {.cell .code execution_count="66"}
``` python
start = time()

vectorizer_di = TfidfVectorizer(stop_words=stopwords, ngram_range=(2,2), max_features=1000000) 
vectorizer_di.fit(x_train_lemmatized)
x_train_tfidf_di = vectorizer_di.transform(x_train_lemmatized)

end = time()

time_tfidf_di = end - start

print("Количество полученных признаков:", x_train_tfidf_di.shape[1])
print("Время генерации:", time_tfidf_di.__round__(1), 'с')
```

::: {.output .stream .stdout}
    Количество полученных признаков: 1000000
    Время генерации: 14.6 с
:::
:::

::: {.cell .markdown}
### Получение эмбедингов с помощью нейросети BERT

Используем данные, очищенные от спец символов, цифр, знаков препинания.\
Не делаем лематизацию и удаление стоп-слов, т.к. это может помешать
нейросети оценить контекст.

Используем предобученную нейросеть специализующуюся на токсичных
комментариях.

Данных много. Прежде, чем перейти к расчёту эмбедингов, оценим примерное
время для выполнения задачи на CPU и GPU.\
**Время GPU существенно дороже, если время выполнения позволяет, то
лучше обойтись CPU**

Сделаем эмбединги для 500 текстовых записей и расчитаем время на
обработку всего корпуса.
:::

::: {.cell .markdown}
#### Оценка времени выполнения задачи на CPU и GPU
:::

::: {.cell .markdown}
CPU
:::

::: {.cell .code execution_count="67"}
``` python
start = time()

embedings_trial = func_get_embedings(x_train[0:500], run_on_gpu=False)

end = time()

time_estimation_embeddings_cpu = round((end - start) * (len(x_train) / 1000) / 60, 0)   
del embedings_trial

print("Расчётное время для получения эмбедингов на CPU:", time_estimation_embeddings_cpu, 'минут')
```

::: {.output .stream .stdout}
    Training will run on CPU.
:::

::: {.output .stream .stderr}
    100%|██████████| 5/5 [04:10<00:00, 50.11s/it]
:::

::: {.output .stream .stdout}
    Расчётное время для получения эмбедингов на CPU: 535.0 минут
:::

::: {.output .stream .stderr}
:::
:::

::: {.cell .markdown}
GPU
:::

::: {.cell .code execution_count="68"}
``` python
start = time()

embedings_trial = func_get_embedings(x_train[0:500], run_on_gpu=True)

end = time()

time_estimation_embeddings_gpu = round((end - start) * (len(x_train) / 1000) / 60, 0)   

del embedings_trial

print("Расчётное время для получения эмбедингов на видеокарте (GPU):", time_estimation_embeddings_gpu, 'минут')
```

::: {.output .stream .stdout}
    GPU: NVIDIA GeForce RTX 3070 Laptop GPU is available. Running on GPU
:::

::: {.output .stream .stderr}
    100%|██████████| 5/5 [00:10<00:00,  2.11s/it]
:::

::: {.output .stream .stdout}
    Расчётное время для получения эмбедингов на видеокарте (GPU): 26.0 минут
:::

::: {.output .stream .stderr}
:::
:::

::: {.cell .markdown}
**Вывод:**

-   Будем работать на GPU, т.к. расчётно потребуется всего ок. 30 мин\
-   Время получения эмбедингов на CPU слишком долгое - в 20 раз больше
    чем на GPU (590+ мин или ок. 10 часов)
:::

::: {.cell .markdown}
#### Получение эмбедингов для обучающей выборки

*Если эмбединги ранее уже были получены на этапе разработки, то загрузим
из файла.*
:::

::: {.cell .code execution_count="69"}
``` python
try:
    x_train_embedded = pd.read_pickle('x_train_embedded.pkl')    
    time_fact_embeddings = 2936.13 # Замерено ранее
    
except:    
    start = time()

    x_train_embedded = func_get_embedings(x_train, run_on_gpu=True)
    x_train_embedded.to_pickle('x_train_embedded.pkl')

    end = time()

    time_fact_embeddings = end - start
```
:::

::: {.cell .markdown}
## Моделирование

Проверим различные сочетания моделей и предобработки данных на
кросс-валидации с подбором гиперпараметров.\
Подбор гиперпараметров может существенно повлиять на точность
Логистической регрессии.
:::

::: {.cell .markdown}
### 1 Логистическая регрессия {#1-логистическая-регрессия}
:::

::: {.cell .markdown}
Зададим единую сетку для подбора гиперпараметров всех подвидов модели.

-   В данных сильный дисбаланс классов: учтём это и поставим настройку
    \'classifier\_\_class_weight\' = \'balanced\'
-   Используем алгоритм \'saga\', он лучше подходят для больших
    датасетов + поддерживает смешанную регуляризацию ElasticNet
-   Для повышения точности модели важно оптимизировать параметры
    регуляризации весов:
    -   Для этого используем смешанную регуляризацию Elasticnet + подбор
        коэф. l1/l2
    -   Инверсию регуляризации (параметр \'C\')

Датасет достаточно большой, а также хочется проверить широкий круг
гиперпараметров.\
Чтобы сэкономить время проведём подбор гиперпараметров в 2 этапа:

-   Сузим круг поиска за счёт предварительного случайного поиска по
    сетке параметров с заданным числом итераций
-   Проведём поиск по более узкой сетке для каждой модели
:::

::: {.cell .markdown}
#### 0 Случайный подбор гиперпараметров Логистической регрессии для сужения сетки поиска {#0-случайный-подбор-гиперпараметров-логистической-регрессии-для-сужения-сетки-поиска}
:::

::: {.cell .markdown}
Исходная широкая сетка гиперпараметров
:::

::: {.cell .code execution_count="70"}
``` python
params_LogReg_long = [{'solver': ['saga'],                # default 'lbfgs'
                       'class_weight': ['balanced'],      # default None
                       'penalty':['elasticnet'],          # default l2
                       'l1_ratio':[0, 0.1, 0.3, 0.7, 1],  # default None
                       'max_iter': [100, 150],            # default 100
                       'C': range(1, 10)                  # default 1.0
                      }]
```
:::

::: {.cell .code execution_count="71"}
``` python
model_LogReg_try = LogisticRegression(random_state=RANDOM_STATE)

grid = RandomizedSearchCV(estimator=model_LogReg_try, param_distributions=params_LogReg_long, 
                          scoring='f1', cv=2, n_iter=25, n_jobs=-1, verbose=0) 
grid.fit(X=x_train_tfidf_uni, y=y_train)

best_params_LogReg_try = grid.best_params_
best_metric_LogReg_try = grid.best_score_

print('--------------------------------------------------------------------------')
print('Лучшая метрика F1 на кросс-валидации:', best_metric_LogReg_try.__round__(3))
print('Лучшие гиперпараметры:', best_params_LogReg_try)
```

::: {.output .stream .stdout}
    --------------------------------------------------------------------------
    Лучшая метрика F1 на кросс-валидации: 0.767
    Лучшие гиперпараметры: {'solver': 'saga', 'penalty': 'elasticnet', 'max_iter': 150, 'l1_ratio': 1, 'class_weight': 'balanced', 'C': 3}
:::

::: {.output .stream .stderr}
    e:\UpdatedProject Venv on Python 3.12.2\.venv\Lib\site-packages\sklearn\linear_model\_sag.py:350: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
:::
:::

::: {.cell .markdown}
Итоговая сокращенная сетка для поиска гиперпараметров
:::

::: {.cell .code execution_count="72"}
``` python
params_LogReg = [{'solver': ['saga'],           # default 'lbfgs'
                  'class_weight': ['balanced'], # default None
                  'penalty':['elasticnet'],     # default l2
                  'l1_ratio':[0.7, 0.8, 0.9],   # default None
                  'max_iter': [150],            # default 100
                  'C': [2, 5, 9]                # default 1.0
                }]
```
:::

::: {.cell .markdown}
#### 1.1 Логистическая регрессия + векторизация по TF-IDF с униграммами {#11-логистическая-регрессия--векторизация-по-tf-idf-с-униграммами}
:::

::: {.cell .code execution_count="73"}
``` python
model_LogReg_tfidf_uni = LogisticRegression(random_state=RANDOM_STATE)

grid = GridSearchCV(estimator=model_LogReg_tfidf_uni, param_grid=params_LogReg, scoring='f1', cv=2, n_jobs=-1, verbose=0) 
grid.fit(X=x_train_tfidf_uni, y=y_train)

best_params_LogReg_tfidf_uni = grid.best_params_
best_metric_LogReg_tfidf_uni = grid.best_score_

print('Лучшая метрика F1 на кросс-валидации:', best_metric_LogReg_tfidf_uni.__round__(3))
print('Лучшие гиперпараметры:', best_params_LogReg_tfidf_uni) 
```

::: {.output .stream .stdout}
    Лучшая метрика F1 на кросс-валидации: 0.772
    Лучшие гиперпараметры: {'C': 2, 'class_weight': 'balanced', 'l1_ratio': 0.7, 'max_iter': 150, 'penalty': 'elasticnet', 'solver': 'saga'}
:::

::: {.output .stream .stderr}
    e:\UpdatedProject Venv on Python 3.12.2\.venv\Lib\site-packages\sklearn\linear_model\_sag.py:350: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
:::
:::

::: {.cell .markdown}
Обучим модель с подобранными гиперпараметрами на всех обучающих данных и
засечём время.
:::

::: {.cell .code execution_count="74"}
``` python
start = time()

best_model_LogReg_tfidf_uni = LogisticRegression(random_state=RANDOM_STATE, **best_params_LogReg_tfidf_uni)
best_model_LogReg_tfidf_uni.fit(x_train_tfidf_uni, y_train)

end = time()

time_fit_LogReg_tfidf_uni = end - start
print('Время обучения:', time_fit_LogReg_tfidf_uni.__round__(3), 'c')
```

::: {.output .stream .stdout}
    Время обучения: 245.536 c
:::

::: {.output .stream .stderr}
    e:\UpdatedProject Venv on Python 3.12.2\.venv\Lib\site-packages\sklearn\linear_model\_sag.py:350: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
:::
:::

::: {.cell .markdown}
#### 1.2 Логистическая регрессия + векторизация по TF-IDF с диграммами {#12-логистическая-регрессия--векторизация-по-tf-idf-с-диграммами}
:::

::: {.cell .code execution_count="75"}
``` python
model_LogReg_tfidf_di = LogisticRegression(random_state=RANDOM_STATE)

grid = GridSearchCV(estimator=model_LogReg_tfidf_di, param_grid=params_LogReg, scoring='f1', cv=2, n_jobs=-1, verbose=0) 
grid.fit(X=x_train_tfidf_di, y=y_train)

best_params_LogReg_tfidf_di = grid.best_params_
best_metric_LogReg_tfidf_di = grid.best_score_

print('Лучшая метрика F1 на кросс-валидации:', best_metric_LogReg_tfidf_di.__round__(3))
print('Лучшие гиперпараметры:', best_params_LogReg_tfidf_di)
```

::: {.output .stream .stdout}
    Лучшая метрика F1 на кросс-валидации: 0.511
    Лучшие гиперпараметры: {'C': 9, 'class_weight': 'balanced', 'l1_ratio': 0.9, 'max_iter': 150, 'penalty': 'elasticnet', 'solver': 'saga'}
:::

::: {.output .stream .stderr}
    e:\UpdatedProject Venv on Python 3.12.2\.venv\Lib\site-packages\sklearn\linear_model\_sag.py:350: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
:::
:::

::: {.cell .markdown}
Обучим модель с подобранными гиперпараметрами на всех обучающих данных и
засечём время.
:::

::: {.cell .code execution_count="76"}
``` python
start = time()

best_model_LogReg_tfidf_di = LogisticRegression(random_state=RANDOM_STATE, **best_params_LogReg_tfidf_di)
best_model_LogReg_tfidf_di.fit(x_train_tfidf_di, y_train)

end = time()

time_fit_LogReg_tfidf_di = end - start
print('Время обучения:', time_fit_LogReg_tfidf_di.__round__(2), 'c')
```

::: {.output .stream .stdout}
    Время обучения: 1424.63 c
:::

::: {.output .stream .stderr}
    e:\UpdatedProject Venv on Python 3.12.2\.venv\Lib\site-packages\sklearn\linear_model\_sag.py:350: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
:::
:::

::: {.cell .markdown}
#### 1.3 Логистическая регрессия + эмбединги от нейросети {#13-логистическая-регрессия--эмбединги-от-нейросети}
:::

::: {.cell .code execution_count="77"}
``` python
model_LogReg_embeddings = LogisticRegression(random_state=RANDOM_STATE)

grid = GridSearchCV(estimator=model_LogReg_embeddings, param_grid=params_LogReg, scoring='f1', cv=2, n_jobs=-1, verbose=0) 
grid.fit(X=x_train_embedded, y=y_train) 

best_params_LogReg_embeddings = grid.best_params_
best_metric_LogReg_embeddings = grid.best_score_

print('Лучшая метрика F1 на кросс-валидации:', best_metric_LogReg_embeddings.__round__(2))
print('Лучшие гиперпараметры:', best_params_LogReg_embeddings)
```

::: {.output .stream .stdout}
    Лучшая метрика F1 на кросс-валидации: 0.89
    Лучшие гиперпараметры: {'C': 5, 'class_weight': 'balanced', 'l1_ratio': 0.8, 'max_iter': 150, 'penalty': 'elasticnet', 'solver': 'saga'}
:::

::: {.output .stream .stderr}
    e:\UpdatedProject Venv on Python 3.12.2\.venv\Lib\site-packages\sklearn\linear_model\_sag.py:350: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
:::
:::

::: {.cell .markdown}
Обучим модель с подобранными гиперпараметрами на всех обучающих данных и
засечём время.
:::

::: {.cell .code execution_count="78"}
``` python
start = time()

best_model_LogReg_embeddings = LogisticRegression(random_state=RANDOM_STATE, **best_params_LogReg_embeddings)
best_model_LogReg_embeddings.fit(x_train_embedded, y_train)

end = time()

time_fit_LogReg_embeddings = end - start
print('Время обучения:', time_fit_LogReg_embeddings.__round__(2), 'c')
```

::: {.output .stream .stdout}
    Время обучения: 116.08 c
:::

::: {.output .stream .stderr}
    e:\UpdatedProject Venv on Python 3.12.2\.venv\Lib\site-packages\sklearn\linear_model\_sag.py:350: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
:::
:::

::: {.cell .markdown}
### 2. Бустинг CatBoost {#2-бустинг-catboost}
:::

::: {.cell .markdown}
В CatBoost встроены токенизатор и методы векторизации, поэтому мало
смысла использовать отдельную векторизацию по TF-IDF.\
Попробуем подать на вход два варианта предобработки:

-   только с базовой очисткой текста
-   эмбединги, полученные от нейросети BERT

Модель хорошо работает из \"коробки\", но достаточно долго обучается.\
По опыту прошлых проектов, подбор гиперпараметров не сильно влияет на
результаты модели.

Перебрём минимальное число гиперпараметров, которые могут немного
улучшить результат (l2 регуляризацию и количество итераций).

*Используем GPU для ускорения работы.*
:::

::: {.cell .code execution_count="79"}
``` python
params_CatBoost = { 'n_estimators': [1000, 1200],     # default 1000 
                    'l2_leaf_reg': [3, 7, 10]         # default 3 
                  }
```
:::

::: {.cell .markdown}
#### 2.1 CatBoost + очищенный текст {#21-catboost--очищенный-текст}
:::

::: {.cell .code execution_count="80"}
``` python
x_train_DF = pd.DataFrame(x_train, columns=['text']) # техническое преобразование в датафрейм для работы модели

model_Cat_clean_text = CatBoostClassifier(random_state=RANDOM_STATE, eval_metric='F1', task_type='GPU')

grid = GridSearchCV(estimator=model_Cat_clean_text, param_grid=params_CatBoost, scoring='f1', cv=2, verbose=0)
grid.fit(X=x_train_DF, y=y_train, text_features=['text'], silent=True)

best_params_Cat_clean_text = grid.best_params_
best_metric_Cat_clean_text = grid.best_score_

print('-------------------------------------')
print()
print('Лучшая метрика F1 на кросс-валидации:', best_metric_Cat_clean_text.__round__(3))
print('Лучшие гиперпараметры:', best_params_Cat_clean_text) 
```

::: {.output .stream .stdout}
    -------------------------------------

    Лучшая метрика F1 на кросс-валидации: 0.773
    Лучшие гиперпараметры: {'l2_leaf_reg': 3, 'n_estimators': 1200}
:::
:::

::: {.cell .markdown}
Обучим модель с подобранными гиперпараметрами на всех обучающих данных и
засечём время.
:::

::: {.cell .code execution_count="81"}
``` python
start = time()

best_model_Cat_clean_text = CatBoostClassifier(random_state=RANDOM_STATE, eval_metric='F1', task_type='GPU', **best_params_Cat_clean_text)
best_model_Cat_clean_text.fit(x_train_DF, y_train, text_features=['text'], silent=True)

end = time()

time_fit_Cat_clean_text = end - start
print('Время обучения:', time_fit_Cat_clean_text.__round__(2), 'c')
```

::: {.output .stream .stdout}
    Время обучения: 18.0 c
:::
:::

::: {.cell .markdown}
#### 2.2 CatBoost + эмбединги от нейросети {#22-catboost--эмбединги-от-нейросети}
:::

::: {.cell .code execution_count="82"}
``` python
model_Cat_embeddings = CatBoostClassifier(random_state=RANDOM_STATE, eval_metric='F1', task_type='GPU')

grid = GridSearchCV(estimator=model_Cat_embeddings, param_grid=params_CatBoost, scoring='f1', cv=2, verbose=0)
grid.fit(X=x_train_embedded, y=y_train, silent=True)

best_params_Cat_embeddings = grid.best_params_
best_metric_Cat_embeddings = grid.best_score_

print('-------------------------------------')
print()
print('Лучшая метрика F1 на кросс-валидации:', best_metric_Cat_embeddings.__round__(3))
print('Лучшие гиперпараметры:', best_params_Cat_embeddings) 
```

::: {.output .stream .stdout}
    -------------------------------------

    Лучшая метрика F1 на кросс-валидации: 0.918
    Лучшие гиперпараметры: {'l2_leaf_reg': 10, 'n_estimators': 1000}
:::
:::

::: {.cell .markdown}
Обучим модель с подобранными гиперпараметрами на всех обучающих данных и
засечём время.
:::

::: {.cell .code execution_count="83"}
``` python
start = time()

best_model_Cat_embeddings = CatBoostClassifier(random_state=RANDOM_STATE, eval_metric='F1', task_type='GPU', **best_params_Cat_embeddings)
best_model_Cat_embeddings.fit(x_train_embedded, y=y_train, silent=True)

end = time()

time_fit_Cat_embeddings = end - start
print('Время обучения:', time_fit_Cat_embeddings.__round__(2), 'c')
```

::: {.output .stream .stdout}
    Время обучения: 15.8 c
:::
:::

::: {.cell .markdown}
### Сводные данные по выбору модели

1.  Наилучшие результаты по метрике показала модель CatBoost с
    использованием эмбедингов от предобученной нейросети.\

-   **Остановим выбор на этой конфигурации**\
-   Однако, стоит отметить, что решение времязатратное и требует
    использования GPU

1.  CatBoost c использованием встроенных средств векторизации показала
    соответсвующую требованиям заказчика точность (0.77 по F1).

-   такое решение гораздо быстрее выбранного, более того самое быстрое
    из рассмотренных при использовании GPU.\
-   Возможно, даже без использования GPU время работы будет достаточно
    оперативным.

1.  Логистическая регрессия показывает хорошие результаты (0.77 - 0.88
    по F1) при использовании эмбедингов и униграмм по TF-IDF.\
    Однако использовать решения на основе данной модели не
    целесообразно, т.к. нет преимуществ по сравнению с бустингом.
:::

::: {.cell .markdown}
Рассчитаем общее время на подготовку всех сочетаний модель +
предобработка к инференсу. Учтём время на:

-   лемматизацию
-   векторизация/получение эмбедингов
-   обучение модели с подобранными

*Очистка данных делается быстро (2-4 c) и используется в любом случае -
можно не учитывать.*
:::

::: {.cell .code execution_count="84"}
``` python
LogReg_tfidf_uni_score = best_metric_LogReg_tfidf_uni.__round__(3)
LogReg_tfidf_uni_time = round((time_lemmatization + time_tfidf_uni + time_fit_LogReg_tfidf_uni) / 60, 1)
uni_text_vec_time = round((time_lemmatization + time_tfidf_uni) / 60, 1)

LogReg_tfidf_di_score = best_metric_LogReg_tfidf_di.__round__(3)
LogReg_tfidf_di_time = round((time_lemmatization + time_tfidf_di + time_fit_LogReg_tfidf_di) / 60, 1)
di_text_vec_time = round((time_lemmatization + time_tfidf_di) / 60, 1)

LogReg_embeddings_score = best_metric_LogReg_embeddings.__round__(3)
LogReg_embeddings_time = round((time_fact_embeddings + time_fit_LogReg_embeddings) / 60, 1)

embeddings_time = round((time_fact_embeddings) / 60, 1)

Cat_clean_text_score = best_metric_Cat_clean_text.__round__(3)
Cat_clean_text_time = round((time_fit_Cat_clean_text) / 60, 1)

Cat_embeddings_score = best_metric_Cat_embeddings.__round__(3)
Cat_embeddings_text_time = round((time_fact_embeddings + time_fit_Cat_embeddings) / 60, 1)
```
:::

::: {.cell .code execution_count="85"}
``` python
score_table = [[LogReg_tfidf_uni_score, LogReg_tfidf_uni_time, uni_text_vec_time],
               [LogReg_tfidf_di_score, LogReg_tfidf_di_time, di_text_vec_time],
               [LogReg_embeddings_score, LogReg_embeddings_time, embeddings_time],
               [Cat_clean_text_score, Cat_clean_text_time, 0],
               [Cat_embeddings_score, Cat_embeddings_text_time, embeddings_time]
                ]

score = pd.DataFrame(index=['LOGISTIC REG УНИГРАММЫ', 'LOGISTIC REG ДИГРАММЫ', 'LOGISTIC REG ЭМБЕДИНГИ', 'CATBOOST ОЧИСТ. ТЕКСТ', 'CATBOOST ЭМБЕДИНГИ'], 
                     data = score_table, columns=['МЕТРИКА НА КРОСС-ВАЛИДАЦИИ', 'ОБЩЕЕ ВРЕМЯ ПОДГОТОВКИ К ИНФЕРЕНС, МИН', 'В Т.Ч. ВРЕМЯ ПРЕОБРАЗОВАНИЯ ТЕКСТОВ, МИН'])

score = score.sort_values(by='МЕТРИКА НА КРОСС-ВАЛИДАЦИИ', ascending=False)

score
```

::: {.output .execute_result execution_count="85"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>МЕТРИКА НА КРОСС-ВАЛИДАЦИИ</th>
      <th>ОБЩЕЕ ВРЕМЯ ПОДГОТОВКИ К ИНФЕРЕНС, МИН</th>
      <th>В Т.Ч. ВРЕМЯ ПРЕОБРАЗОВАНИЯ ТЕКСТОВ, МИН</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CATBOOST ЭМБЕДИНГИ</th>
      <td>0.918</td>
      <td>49.2</td>
      <td>48.9</td>
    </tr>
    <tr>
      <th>LOGISTIC REG ЭМБЕДИНГИ</th>
      <td>0.888</td>
      <td>50.9</td>
      <td>48.9</td>
    </tr>
    <tr>
      <th>CATBOOST ОЧИСТ. ТЕКСТ</th>
      <td>0.773</td>
      <td>0.3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>LOGISTIC REG УНИГРАММЫ</th>
      <td>0.772</td>
      <td>4.8</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>LOGISTIC REG ДИГРАММЫ</th>
      <td>0.511</td>
      <td>24.6</td>
      <td>0.8</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
### Сводные данные по выбору модели {#сводные-данные-по-выбору-модели}

1.  Наилучшие результаты по метрике показала модель CatBoost с
    использованием эмбедингов от предобученной нейросети.\

-   **Остановим выбор на этой конфигурации**\
-   Однако, стоит отметить, что решение времязатратное и требует
    использования GPU

1.  CatBoost c использованием встроенных средств векторизации текста
    показала точность (0.77 по F1) соответсвующую требованиям заказчика.

-   такое решение гораздо быстрее выбранного, более того, самое быстрое
    из рассмотренных при использовании GPU.\
-   Возможно, даже без использования GPU время работы будет достаточно
    оперативным.

1.  Логистическая регрессия показывает хорошие результаты (0.77 - 0.88
    по F1) при использовании эмбедингов и униграмм по TF-IDF.\
    Однако использовать решения на основе данной модели не
    целесообразно, т.к. нет преимуществ по сравнению с бустингом.
:::

::: {.cell .markdown}
## Тестирование

Проверим модель на адекватность: сравним с бейзлайном - \"дамми\"
моделью, предсказывающей исходя из распределения ответов в датасете.\
Сделаем \"дамми\" модель на тестовых данных, чтобы имитировать вновь
поступающие данные.
:::

::: {.cell .markdown}
Бэйзлан для проверки модели
:::

::: {.cell .code execution_count="86"}
``` python
dummy_model = DummyClassifier (strategy='stratified', random_state=RANDOM_STATE)
dummy_model.fit(x_test, y_test)
preds = dummy_model.predict(x_test)

dummy_test_score = round(f1_score(y_true=y_test, y_pred=preds), 3)

print('Метрика дамми-модели:', dummy_test_score)
```

::: {.output .stream .stdout}
    Метрика дамми-модели: 0.104
:::
:::

::: {.cell .markdown}
Для получения предсказаний сгенирируем эмбединги для тестовых данных
:::

::: {.cell .code execution_count="87"}
``` python
x_test_embedded = func_get_embedings(x_test, run_on_gpu=True)
```

::: {.output .stream .stdout}
    GPU: NVIDIA GeForce RTX 3070 Laptop GPU is available. Running on GPU
:::

::: {.output .stream .stderr}
    100%|██████████| 777/777 [11:35<00:00,  1.12it/s]
:::
:::

::: {.cell .markdown}
Итоговая метрика лучшей модели
:::

::: {.cell .code execution_count="88"}
``` python
final_preds = best_model_Cat_embeddings.predict(data=x_test_embedded)
final_f1_score = f1_score(y_true=y_test, y_pred=final_preds)

print('Итоговая метрика лучшей модели на тестовых даннных:', round(final_f1_score, 3))
```

::: {.output .stream .stdout}
    Итоговая метрика лучшей модели на тестовых даннных: 0.911
:::
:::

::: {.cell .markdown}
## Выводы проекта
:::

::: {.cell .markdown}
**Данную задачу классификации эмоциональной окраски текстов оптимально
решать при помощи:**

1.  Эмбедингов от специализированной предобученной нейросети + модели
    CatBoost; если доступна работа на GPU:\

-   Решение позволяет получить очень высокую точность (0.9+ по метрике
    F1) и превосходит требования заказчика (0.75+)

-   Ключевым является применение GPU - т.к. без этого ресурса время
    подготовки к инференс может увеличится в 15-20 раз

-   Заказчику необходимо подсветить, что даже с использованием GPU
    подготовка к инференс занимает длительное время.\
    Это может создать трудности, если требуется часто переобучать
    модель.

1.  Модели CatBoost на текстах с минимальной предобработкой (очисткой).\
    Это хороший вариант при ограниченном времени GPU и необходимости
    частого переобучения модели.

-   Такое решение является самым быстрым из рассмотренных

-   Обеспечивает приемлемый уровень точности (0.77+ по метрике F1),
    соответсвующий требованиям заказчика

-   Решение является самым надёжным, т.к.:

    -   Оно простое, не требует отдельного этапа предобработки
    -   Не требует использования сторонней предобученной на определённой
        тематике нейросети

1.  Использование логистической регрессии и данных предобработанных по
    технологии TF-IDF не целесообразно.\
    Данная схема не показала конкурентных результатов ни по времении, ни
    по точности.\
    Возможно, у этой схемы есть шанс при работе только на CPU, если
    подготовка к инференс по варианту 2. будет занимать длительное
    время.

**Возможные пути дальнейшего улучшения решения:**

-   сделать тонкую предобработку текстов прописав отдельный аглоритм на
    правилах на основе глубокого анализа текстов
-   более тонкий подбор гиперпараметров итоговой модели

**Необходимо обсудить варианты 1. и 2. с заказчиком и выяснить
возможность выделения времени на для работы на GPU**
:::
