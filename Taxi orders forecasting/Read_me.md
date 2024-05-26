#  Прогнозирование заказов такси

Задача срогнозировать количество заказов такси на следующий час. 
 
Заказчик предоставил данные за период ок. 6 мес.  
Точность прогнозирования проверяется по метрике RMSE и должна быть не выше 48 на тестовой выборке в 10%.  

[ipynb](https://github.com/sotwra/Portfolio/blob/main/Taxi%20orders%20forecasting/Forecasting_Taxi.ipynb))

## Стек

- python
- pandas
- numpy
- sklearn
- statsmodels

### Анализ данных date time
- statsmodels.tsa.seasonal

### Модели
- Prophet
- LGBMRegressor (lightgbm)
- LinearRegression, ElasticNet, DummyRegressor (sklearn)

### Вспомогательные средсва и визуализация
- GridSearchCV, TimeSeriesSplit (sklearn)
- Отслеживание времени выполнения (tqdm, time)
- itertools
- cross_validation, performance_metrics (prophet)
- matplotlib *(Для Git Hub не использую 'тяжелую' графику, например, cufflinks/plotly)*

## Выводы

Исследованы точности и время обучения на CPU моделей ниже.  
Проведён подбор гиперпараметров на кросс-валидации.
- Универсальные ML модели + генерация кастомных синтетических признаков
- Специализированная модель Prophet

1. Наиболее точное и быстрое решение показала Линейная регрессия.
  - Только эта модель соотвествует требованиям заказчика по точности.
  - Возможно данных недостаточно для раскрытия потенциала других рассмотренных моделей. 
    На это указывает, что на тесте результаты моделей до 2,5 раза хуже чем на кросс-валидации.  

2. Близкую к требуемой точность показал Prophet.  
   Модель удобна, т.к. автоматически генерирует набор признаков + имеет хорошо интерпретируемые гиперпараметры.

| Анализ данных date time | Модели | Вспомогательные средсва и визуализация |
| ----------------------- | ------ | -------------------------------------- |
| Statsmodels | - Prophet <br/> - LGBMRegressor <br/> - LinearRegression, ElasticNet, DummyRegressor (sklearn) | - GridSearchCV, TimeSeriesSplit (sklearn) <br/> - Отслеживание времени выполнения (tqdm, time) <br/> - itertools <br/>  - cross_validation, performance_metrics (prophet) <br/> - matplotlib  
