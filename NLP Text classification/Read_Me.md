# Анализ текстов на токсичность
[Ссылка на тетрадку в формате ipynb](https://github.com/sotwra/Portfolio/blob/main/NLP%20Text%20classification/NLP%20comments%20classification.ipynb)

## Стек
- python
- pandas
- numpy
- sklearn

| Обработка текста | Модели | Вспомогательные средсва |
| ---------------- | ------ | ----------------------- |
| BertTokenizer, BertModel (transformers) <br/> TfidfVectorizer (sklearn) <br/> Регулярные выражения (re) <br/> SnowballStemmer, stopwwords (nltk) | CatBoostClassifier <br/> LogisticRegression (sklearn) <br/> DummyClassifier | GridSearchCV, RandomizedSearchCV (sklearn) <br/> Время выполнения (tqdm, time)

## Выводы
Проведено исследование вариантов сочетаний предобработки текстов и ML моделей.   
Оценено время подготовки к инференс и необходимость GPU. 
1. Наиболее точное решение на основе эмбедингов от спец. предобученной нейросети + модели CatBoost.  
Однако решение требует достаточно много времени для подготовки к инференс и наличия GPU.
  
3. Приемлемую точность у модели CatBoost с использованием встроенной векторизации текстов.   
При этом меньше требования к оборудованию и время подготовки к инференс.
