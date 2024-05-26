::: {.cell .markdown}
# Прогнозирование заказов такси
:::

::: {.cell .markdown}
Сервис заказа такси собрал исторические данные о заказах такси в
аэропортах.\
Компания хочет привлекать больше водителей в период пиковой нагрузки для
удовлетворения дополнительного спроса.

**Задача**

Срогнозировать количество заказов такси на следующий час.\
Точность прогнозирования проверяется по метрике RMSE и должна быть не
выше 48 на тестовой выборке в 10%.
:::

::: {.cell .markdown}
## План работы

1.  Загрузить и исследовать данные

2.  Проанализировать данные на наличие тренда и сезонности

3.  Обучить модели с подбором оптимального набора признаков + подбором
    гиперпараметров на кросс-валидации:

    -   Линейная регрессия
    -   регрессия Elastic Net
    -   LightGBM бустинг
    -   Prophet
    -   Сделать сводую таблицу результатов кросс-валидации

4.  Провести тестирование на 10% данных

    -   Сгенерировать \"дамми\" модель для проверки моделей на
        адекватность
    -   Проверить модели *(i)* на тестовых данных

5.  Составить выводы и рекомендации

*(i) В виду специфики проекта отойдём от классического подхода и
проверим все модели*
:::

::: {.cell .markdown}
## Описание данных

Данные находятся в файле `taxi.csv`.\
Имеется столбец со временем заказа и кол-во заказов в столбце
`num_orders`.
:::

::: {.cell .markdown}
## Подготовка и анализ данных
:::

::: {.cell .markdown}
### Подключение библиотек
:::

::: {.cell .code execution_count="1"}
``` python
import numpy as np
import pandas as pd

#Прогресс выполнения и время работы
from time import time
from tqdm import tqdm

#Модели
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from prophet import Prophet
from sklearn.dummy import DummyRegressor

# Метрики, средства анализа, прочее
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import itertools
from prophet.diagnostics import cross_validation, performance_metrics

# Визуализация
# Для Git Hub не использую 'тяжелую' графику, например, cufflinks/plotly
import matplotlib.pyplot as plt
```
:::

::: {.cell .markdown}
### Настройки и константы
:::

::: {.cell .code execution_count="2"}
``` python
RANDOM_STATE = 12345

# Выключим назойливые логи-простыни

from warnings import simplefilter 
simplefilter (action='ignore', category=pd.errors.PerformanceWarning)

import sklearn.exceptions as se 
simplefilter (action='ignore', category=se.FitFailedWarning)
simplefilter (action='ignore', category=se.ConvergenceWarning)

import logging

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled=True
logging.getLogger('fbprophet').disabled = True
```
:::

::: {.cell .markdown}
### Пользовательские функции
:::

::: {.cell .markdown}
#### Генерация признаков из данных о дате и времени

Для ML моделей общего назначения требуются синтетические признаки.\
Можно использовать параметры их генерации для минимизации ошибки
предсказания конкретной ML модели.

Генерируемые признаки

-   Неизменяемые
    -   день недели в виде числа
    -   час
    -   разница значений предыдущий-текущий период
-   Изменяемые в зависимости от настроек генерации
    -   скользящее среднее с варьируемой шириной \"окна\"
    -   скользящее стандартное отклонение с варьируемой шириной \"окна\"
    -   отстающие значения (\"лаги\"), варьируемое количество

*Месяц использовать некорректно, т.к. не достаточно данных (нужны данные
хотя бы за 12 мес.)*
:::

::: {.cell .code execution_count="3"}
``` python
def func_make_features(data, max_lag, rolling_size):
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    data['hour'] = data.index.hour

    # Изменяемые и расчётные признаки

    #Текущее значение ряда для расчёта скользящего среднего, ст. отклонения и разницы значенений, применять нельзя.
    #Это утечка целевого признака. Shift убирает текущее значение.

    # Разница значений прошлых периодов
    data['diff'] = data['num_orders'].shift().fillna(0) - data['num_orders'].shift(2).fillna(0)

    # Скользящее среднее
    data['rolling_mean'] = data['num_orders'].shift().rolling(rolling_size).mean().fillna(0)

    # Cтандартное отклонение
    data['rolling_std'] = data['num_orders'].shift().rolling(rolling_size).std().fillna(0)

    # Лаги
    # Пустоты заполним нулями
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag).fillna(0)

    # Первые 2 строки не информативны, т.к. состоят почти полностью из нулей
    data.drop(index=data.index[0:1], inplace=True)
```
:::

::: {.cell .markdown}
#### Подбор оптимального числа лагов и ширины скользящего окна

-   возращает оптимизированный под конкретную ML модель датасет
-   для ускорения работы прекращает перебор по \"множителю отсечки\"
    если нет улучшения уже достигнутой метрики
-   строит графики подбора
-   можно задавать \"множитель отсечки\" и лимит для перебора

*Подбирает число лагов, а потом к нему ширину окна. Такой подход
оказался более быстрым по сравнению с вложенными циклами при такой же
точности.*
:::

::: {.cell .code execution_count="4"}
``` python
def func_pick_lags_rolls(model:object, df_data:object, show_plot=False, max_pick=1000, pick_break_limiter=1.12):

    # Инициируем начальные значения
    best_rmse = 100
    best_lags = 1
    best_roll = 1
    # тут логи и RMSE для графика
    log_lags = [[], []] 
    log_rolls = [[], []]

    # ПОДБИРАЕМ ОПТИМАЛЬНОЕ ЧИСЛО ЛАГОВ, ЗАТЕМ К НЕМ ОПТИМАЛЬНОЕ ОКНО СКОЛЬЗЯЩИХ ЗНАЧЕНИЙ
    # Перебираем лаги по 6 часов
    for lag in tqdm(range(6, max_pick, 6)):
        data = df_data.copy()
        func_make_features(data, max_lag=lag, rolling_size=12)   
        train, valid = train_test_split(data, test_size=0.2, shuffle=False)

        x_train = train.drop(columns='num_orders')
        y_train = train['num_orders']
        x_valid = valid.drop(columns='num_orders')
        y_valid = valid['num_orders']

        model.fit(x_train, y_train)
        preds = model.predict(x_valid)
        rmse = mean_squared_error(preds, y_valid)**0.5
        log_lags[0].append(lag)
        log_lags[1].append(rmse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_lags = lag
        if rmse > best_rmse * pick_break_limiter:
            break  

    # Перебираем ширину окон по 12 часов
    # Инициируем начальные значения
    best_rmse = 100   
    for roll_size in tqdm(range(6, 73, 6)):
        data = df_data.copy()
        func_make_features(data, max_lag=best_lags, rolling_size=roll_size)   
        train, valid = train_test_split(data, test_size=0.2, shuffle=False)

        x_train = train.drop(columns='num_orders')
        y_train = train['num_orders']
        x_valid = valid.drop(columns='num_orders')
        y_valid = valid['num_orders']

        model.fit(x_train, y_train)
        preds = model.predict(x_valid)
        rmse = mean_squared_error(preds, y_valid)**0.5
        log_rolls[0].append(roll_size)
        log_rolls[1].append(rmse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_roll = roll_size
    
    # ЗАПИШЕМ ИТОГОВЫЕ ПОКАЗАТЕЛИ
    print()
    print('------------------------------------')
    print('Наилучшая метрика', best_rmse.__round__(3))
    print('Оптимальное число лагов в признаках', best_lags)
    print('Оптимальная ширина окна', best_roll)

    # СТРОИМ ГРАФИКИ
    if show_plot:       
        pd.DataFrame({'Число лагов':log_lags[0], 'Метрика RMSE':log_lags[1]}).plot(
                       title='Зависимость метрики от числа лагов', 
                       y='Метрика RMSE', x='Число лагов', 
                       xlabel='число лагов', ylabel='RMSE', 
                       legend=False, figsize=(10,3))
        
        pd.DataFrame({'Окно':log_rolls[0], 'Метрика RMSE':log_rolls[1]}).plot(
                       title='Зависимость метрики от ширины окон среднего значения и ст. отклонения', 
                       y='Метрика RMSE', x='Окно', 
                       xlabel='ширина окна', ylabel='RMSE', 
                       legend=False, figsize=(10,3));    

    return best_lags, best_roll
```
:::

::: {.cell .markdown}
### Загрузка и знакомство с данными
:::

::: {.cell .code execution_count="5"}
``` python
# Загрузка данных

try:
    df = pd.read_csv("taxi.csv", index_col=[0], parse_dates=[0])
except:
    df = pd.read_csv('/datasets/taxi.csv', index_col=[0], parse_dates=[0])
```
:::

::: {.cell .code execution_count="6"}
``` python
# Проверка загрузки и парсинга дат в индекс 

df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 26496 entries, 2018-03-01 00:00:00 to 2018-08-31 23:50:00
    Data columns (total 1 columns):
     #   Column      Non-Null Count  Dtype
    ---  ------      --------------  -----
     0   num_orders  26496 non-null  int64
    dtypes: int64(1)
    memory usage: 414.0 KB
:::
:::

::: {.cell .code execution_count="7"}
``` python
df.head()
```

::: {.output .execute_result execution_count="7"}
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
      <th>num_orders</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-03-01 00:00:00</th>
      <td>9</td>
    </tr>
    <tr>
      <th>2018-03-01 00:10:00</th>
      <td>14</td>
    </tr>
    <tr>
      <th>2018-03-01 00:20:00</th>
      <td>28</td>
    </tr>
    <tr>
      <th>2018-03-01 00:30:00</th>
      <td>20</td>
    </tr>
    <tr>
      <th>2018-03-01 00:40:00</th>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
Задача прогнозировать кол-во заказов в час, по этому сделаем ресемплинг
по одному часу.
:::

::: {.cell .code execution_count="8"}
``` python
df = df.resample('1h').sum()
df_1h_raw = df.copy()
```
:::

::: {.cell .code execution_count="9"}
``` python
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 4416 entries, 2018-03-01 00:00:00 to 2018-08-31 23:00:00
    Freq: h
    Data columns (total 1 columns):
     #   Column      Non-Null Count  Dtype
    ---  ------      --------------  -----
     0   num_orders  4416 non-null   int64
    dtypes: int64(1)
    memory usage: 69.0 KB
:::
:::

::: {.cell .markdown}
**Выводы первичного анализа данных**

-   данные за 6 полных месяцев с марта по август 2018 включительно;
    признак месяца использовать не целесообразно.
-   пропусков нет
-   данные даны с интервалом 10 мин, проведён ресемплинг по часу
-   данных немного - 4.4 тыс. строк после ресемплинга
:::

::: {.cell .markdown}
### Анализ
:::

::: {.cell .markdown}
Отсортируем данные по времени и верхнеуровнево посмотрим
:::

::: {.cell .code execution_count="10"}
``` python
df.sort_index(ascending=True, inplace=True)
```
:::

::: {.cell .code execution_count="11"}
``` python
df.plot(figsize=(15,5), legend=False, title='Количество заказов такси в час');
```

::: {.output .display_data}
![](Taxi orders forecasting/Forecasting_Taxi_Picks/d9ca1abe592342d27d0a07b438491762bae070a4.png)
:::
:::

::: {.cell .markdown}
График с ресемплированием в час не очень хорошо читается, сделаем
аналогичный по дням.
:::

::: {.cell .code execution_count="12"}
``` python
df.resample('1d').sum().plot(figsize=(15,5), legend=False, title='Количество заказов такси в день');
```

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/f00de13f643eefc7435ad7b9724aa42e3fe31be6.png)
:::
:::

::: {.cell .markdown}
Видно, что дела у компании идут в гору.

-   Есть явный восходящий тренд, ускоряющийся в июне.\
-   Из данных по заказам в час и в день видно, что есть пиковые значения
    в разы превышаюшие средние.
:::

::: {.cell .code execution_count="13"}
``` python
df.boxplot(vert=False, figsize=(15,5));
plt.title('Данные о распредении заказов в час')
plt.xticks(range(0,500,15));
```

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/761712086ccc38aaa8dabfb375e37597e44408a2.png)
:::
:::

::: {.cell .markdown}
-   Медианное значение количества заказов в час 75
-   Типичное количество заказов в час находится в диапазоне от 0 до 180\
-   Есть пиковые значения (выбросы) от 185 до 465
:::

::: {.cell .markdown}
Рассмотрим тренд и сезонность в данных
:::

::: {.cell .code execution_count="14"}
``` python
decompose = seasonal_decompose(df)
trend = decompose.trend
seasonal = decompose.seasonal
residue = decompose.resid
```
:::

::: {.cell .markdown}
#### Тренд
:::

::: {.cell .code execution_count="15"}
``` python
trend.plot(figsize=(15,5), title='Тренд');
```

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/bf970058c408217916bfccff5b92f89d5abab2bf.png)
:::
:::

::: {.cell .markdown}
Подтверждается, что имеется тренд на увеличение заказов в час,
ускоряющийся начиная с июня.
:::

::: {.cell .markdown}
#### Часовая сезонность

Ниже 3 графика с интервалами по 2 дня в начале, середине и конце
датасета, чтобы был виден каждый час.\
Благодаря этому можно посмотреть сезонность по часам.
:::

::: {.cell .code execution_count="16"}
``` python
seasonal['2018-03-01':'2018-03-02'].plot(figsize=(15,5), title='Часовая сезонность в марте (начало датасета)');
```

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/8d3a50835f65f963cf56f8d4b6c06f8bec0a1232.png)
:::
:::

::: {.cell .code execution_count="17"}
``` python
seasonal['2018-05-30':'2018-05-31'].plot(figsize=(15,5), title='Часовая сезонность в конце мая (середина датасета)');
```

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/3fd1346c3feeefb094f780593ae1781818213e5b.png)
:::
:::

::: {.cell .code execution_count="18"}
``` python
seasonal['2018-08-29':'2018-08-30'].plot(figsize=(15,5), title='Часовая сезонность в конце августа (конец датасета)');
```

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/26ce308d28b660a56083ab5d1e9aeed6bd57a8b3.png)
:::
:::

::: {.cell .markdown}
-   Есть устойчивая сезонность в зависимости от часа стабильная
    независимо от месяца:
    -   пики в полночь и ок. 13 и 17 часов, а так же в полночь
    -   спад ок. 6 часов утра.
:::

::: {.cell .markdown}
#### Недельная сезонность

Посмотрим есть ли зависимость числа заказов от дня недели
:::

::: {.cell .code execution_count="19"}
``` python
df_week_day = df.resample('1D').sum().reset_index()
df_week_day['day_of_week'] = df_week_day['datetime'].dt.dayofweek
df_week_day.set_index('datetime', inplace=True)
```
:::

::: {.cell .code execution_count="20"}
``` python
by_day_of_week = df_week_day.groupby(by='day_of_week').agg('sum')
```
:::

::: {.cell .code execution_count="21"}
``` python
by_day_of_week.plot(figsize=(15,5), legend=False, title='ЗАВИСИМОСТЬ ЧИСЛА ЗАКАЗОВ ОТ ДНЯ НЕДЕЛИ');
plt.xticks(labels=['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс'], ticks=range(0,7));
```

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/d7b0fd78464f21e7363ef94f81c1a3b074ed3939.png)
:::
:::

::: {.cell .markdown}
#### Выводы анализа

-   Есть пиковые значения числа заказов, до 6 раз превышающие медианное
    значение (75)
-   Типичное количество заказов в час находится в диапазоне от 0 до 180\
-   Данные нестационарные, т.к.:
    -   Есть устойчивая внутрисуточная сезонность независимо от месяца:
        -   пики в полночь и ок. 13 и 17 часов, а так же в полночь
        -   спад ок. 6 часов утра.
    -   Наблюдается умеренный восходящий тренд до июня, затем рост
        ускоряется в середине периода
-   Из-за ускорения темпов роста в июле среднее число заказов так же
    будет увеличиваться
-   В разные дни недели число заказов существенно различается:
    -   понедельник и пятница наиболее загруженные дни
    -   вторник и воскресенье наименее
:::

::: {.cell .markdown}
## Моделирование
:::

::: {.cell .markdown}
### О создании признаков для моделирования

-   Условия генерации признаков можно связать с минимизацией ошибки
    предсказания конкретной ML модели.\

-   Реализуем генерацию признаков c помощью приведённых выше
    пользовательских функций.\
    Такое решение обеспечит гибкость на инференсе, т.к. позволит при
    каждом обучении модели автоматически подстваиваться под возможные
    изменения в тренде и сезонности.

-   Для модели Prophet требуется отдельная предобработка данных.
:::

::: {.cell .markdown}
### Разбиение на обучающую и тестовую выборки

Важно, чтобы тестовые данные строго были за период после обучающих.
Иначе произойдёт утечка целевого признака.\
Для разбивки используем классический метод, но без перемешивания.

Разбивку сделаем на 3 выборки:

-   обучающую
-   валидационную (для подбора параметров генерации признаков)
-   тестовую в 10% данных для итогового тестирования согласно требований
    заказчика
:::

::: {.cell .code execution_count="22"}
``` python
# Убедимся, что данные отсортированы по времени
df.sort_index(ascending=True, inplace=True)

# Разобьем данные без перемешивания, чтобы сохранить направленность временного ряда
df_train_valid, df_test = train_test_split(df, test_size=0.1, shuffle=False)
```
:::

::: {.cell .markdown}
### Линейная регрессия
:::

::: {.cell .markdown}
##### Подбор числа лагов и ширины окон для скользащего среднего и ст. отклонения {#подбор-числа-лагов-и-ширины-окон-для-скользащего-среднего-и-ст-отклонения}
:::

::: {.cell .code execution_count="23"}
``` python
best_LR_lag, best_LR_roll = func_pick_lags_rolls(LinearRegression(), df_data=df_train_valid, show_plot=True)
```

::: {.output .stream .stderr}
     95%|█████████▌| 158/166 [00:53<00:02,  2.94it/s]
    100%|██████████| 12/12 [00:02<00:00,  4.68it/s]
:::

::: {.output .stream .stdout}

    ------------------------------------
    Наилучшая метрика 27.038
    Оптимальное число лагов в признаках 336
    Оптимальная ширина окна 12
:::

::: {.output .stream .stderr}
:::

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/c57360f096db4b9e3aa70dfb1e68a2669e890b6e.png)
:::

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/d6f55ea6aa9518f0c88a31e30f6833bb179d60db.png)
:::
:::

::: {.cell .markdown}
##### Подбор гиперпараметров модели при оптимальных показателях генерации признаков

Оптимизируем метрику через подбор гиперпараметров на кросс-валидации:

-   считать ли свободный коэффициент регрессии (интерсепт)
    (`fit_intercept`)
:::

::: {.cell .code execution_count="24"}
``` python
# После изысканий выше знаем оптимальное число лагов и величины окон.
# Сгенерируем финальный датасет для подбора гиперпараметров и замера времени обучения.

data_LR = df_train_valid.copy()
func_make_features(data_LR, max_lag=best_LR_lag, rolling_size=best_LR_roll)
x_LR = data_LR.drop(columns='num_orders')
y_LR = data_LR['num_orders']
```
:::

::: {.cell .code execution_count="25"}
``` python
model = LinearRegression()
params = {'fit_intercept':[True, False]} # default True

grid_LR = GridSearchCV(estimator=model,
                            param_grid=params,
                            cv=TimeSeriesSplit(n_splits=3).split(x_LR, y_LR),
                            scoring='neg_mean_squared_error') 

grid_LR.fit(x_LR, y_LR)
```

::: {.output .execute_result execution_count="25"}
```{=html}
<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=&lt;generator object TimeSeriesSplit.split at 0x00000287A1410CA0&gt;,
             estimator=LinearRegression(),
             param_grid={&#x27;fit_intercept&#x27;: [True, False]},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=&lt;generator object TimeSeriesSplit.split at 0x00000287A1410CA0&gt;,
             estimator=LinearRegression(),
             param_grid={&#x27;fit_intercept&#x27;: [True, False]},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">estimator: LinearRegression</label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LinearRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div></div></div></div></div></div>
```
:::
:::

::: {.cell .code execution_count="26"}
``` python
print('Оптимальные гиперпараметры модели', grid_LR.best_params_)
print('Наилучшая метрика на кросс-валидации:', round((grid_LR.best_score_ * -1) ** 0.5, 3))

# Сохраним итоговый результат для сводной таблицы 
best_LR_rmse = (grid_LR.best_score_ * -1) ** 0.5
```

::: {.output .stream .stdout}
    Оптимальные гиперпараметры модели {'fit_intercept': False}
    Наилучшая метрика на кросс-валидации: 24.641
:::
:::

::: {.cell .markdown}
##### Время обучения

Обучим модель с подобранными гипераметрами на обучающей и валидационной
выборках.\
Засечём время обучения.
:::

::: {.cell .code execution_count="27"}
``` python
LR_model_best = LinearRegression(**grid_LR.best_params_)
data = df_train_valid.copy()
func_make_features(data, max_lag=best_LR_lag, rolling_size=best_LR_roll)
x = data.drop(columns='num_orders')
y = data['num_orders']

start = time()
LR_model_best.fit(x, y)
end = time()

LR_time = round((end - start), 2)

print('Время обучения модели', LR_time)
```

::: {.output .stream .stdout}
    Время обучения модели 0.1
:::
:::

::: {.cell .markdown}
### Регрессия ElasticNet

Данная модель - вариант линейной регрессии с гибридной l1/l2
регуляризацией.
:::

::: {.cell .markdown}
##### Подбор числа лагов и ширины окон для скользащего среднего и ст. отклонения {#подбор-числа-лагов-и-ширины-окон-для-скользащего-среднего-и-ст-отклонения}
:::

::: {.cell .code execution_count="28"}
``` python
best_Elastic_lag, best_Elastic_roll = func_pick_lags_rolls(ElasticNet(random_state=RANDOM_STATE), df_data=df_train_valid, show_plot=True)
```

::: {.output .stream .stderr}
    100%|██████████| 166/166 [02:48<00:00,  1.01s/it]
    100%|██████████| 12/12 [00:05<00:00,  2.03it/s]
:::

::: {.output .stream .stdout}

    ------------------------------------
    Наилучшая метрика 26.99
    Оптимальное число лагов в признаках 336
    Оптимальная ширина окна 48
:::

::: {.output .stream .stderr}
:::

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/dc6523d83abb7d85cabd5ce15e560e54e18dd97e.png)
:::

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/509c0088613e793f56990c9052b6d860b3cba7c8.png)
:::
:::

::: {.cell .markdown}
#### Подбор гиперпараметров

Оптимизируем метрику через подбор гиперпараметров на кросс-валидации:

-   соотношение l1/l2 регуляризации (`l1_ratio`)
-   считать ли свободный коэффициент регрессии (интерсепт)
    (`fit_intercept`)
-   ставить ли принудительно только положительные веса (`positive`)
:::

::: {.cell .code execution_count="29"}
``` python
# После изысканий выше знаем оптимальное число лагов и величины окон для регрессии
# Сделаем из них датасет для подбора гиперпараметров и замера временени обучения

data = df_train_valid.copy()
func_make_features(data, max_lag=best_Elastic_lag, rolling_size=best_Elastic_roll)
x = data.drop(columns='num_orders')
y = data['num_orders']
```
:::

::: {.cell .code execution_count="30"}
``` python
model = ElasticNet(random_state=RANDOM_STATE)
params = {'l1_ratio':np.arange(0, 1.1, 0.1),  # default 0.5
          'fit_intercept':[True, False],      # default True
          'positive':[True, False],           # default False
          'max_iter':[1000, 2000]             # default 1000
          } 

grid_Elastic = GridSearchCV(estimator=model,
                            param_grid=params,
                            cv=TimeSeriesSplit(n_splits=3).split(x),
                            scoring='neg_mean_squared_error')

grid_Elastic.fit(x, y)
```

::: {.output .execute_result execution_count="30"}
```{=html}
<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=&lt;generator object TimeSeriesSplit.split at 0x00000287A1410F60&gt;,
             estimator=ElasticNet(random_state=12345),
             param_grid={&#x27;fit_intercept&#x27;: [True, False],
                         &#x27;l1_ratio&#x27;: array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                         &#x27;max_iter&#x27;: [1000, 2000], &#x27;positive&#x27;: [True, False]},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=&lt;generator object TimeSeriesSplit.split at 0x00000287A1410F60&gt;,
             estimator=ElasticNet(random_state=12345),
             param_grid={&#x27;fit_intercept&#x27;: [True, False],
                         &#x27;l1_ratio&#x27;: array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                         &#x27;max_iter&#x27;: [1000, 2000], &#x27;positive&#x27;: [True, False]},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">estimator: ElasticNet</label><div class="sk-toggleable__content fitted"><pre>ElasticNet(random_state=12345)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;ElasticNet<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.ElasticNet.html">?<span>Documentation for ElasticNet</span></a></label><div class="sk-toggleable__content fitted"><pre>ElasticNet(random_state=12345)</pre></div> </div></div></div></div></div></div></div></div></div>
```
:::
:::

::: {.cell .code execution_count="31"}
``` python
print('Оптимальные гиперпараметры модели', grid_Elastic.best_params_)
print('Наилучшая метрика:', round((grid_Elastic.best_score_ * -1) ** 0.5, 3))

# Сохраним итоговый результат для сводной таблицы 
best_Elastic_rmse = (grid_Elastic.best_score_ * -1) ** 0.5
```

::: {.output .stream .stdout}
    Оптимальные гиперпараметры модели {'fit_intercept': False, 'l1_ratio': 1.0, 'max_iter': 1000, 'positive': True}
    Наилучшая метрика: 23.188
:::
:::

::: {.cell .markdown}
##### Время обучения {#время-обучения}

Обучим модель с подобранными гипераметрами на обучающей и валидационной
выборках.\
Засечём время обучения.
:::

::: {.cell .code execution_count="32"}
``` python
Elastic_model_best = ElasticNet(random_state=RANDOM_STATE, **grid_Elastic.best_params_)

start = time()
Elastic_model_best.fit(x, y)
end = time()

Elastic_time = round((end - start), 2)

print('Время обучения модели', Elastic_time)
```

::: {.output .stream .stdout}
    Время обучения модели 0.1
:::
:::

::: {.cell .markdown}
### LightGBM
:::

::: {.cell .markdown}
##### Подбор числа лагов и ширины окон для скользащего среднего и ст. отклонения {#подбор-числа-лагов-и-ширины-окон-для-скользащего-среднего-и-ст-отклонения}
:::

::: {.cell .code execution_count="33"}
``` python
best_LGBM_lag, best_LGBM_roll = func_pick_lags_rolls(LGBMRegressor(random_state=RANDOM_STATE, device='GPU', verbose=-1), df_data=df_train_valid, show_plot=True)
```

::: {.output .stream .stderr}
      0%|          | 0/166 [00:00<?, ?it/s]e:\UpdatedProject Venv on Python 3.12.2\.venv\Lib\site-packages\joblib\externals\loky\backend\context.py:136: UserWarning: Could not find the number of physical cores for the following reason:
    found 0 physical cores < 1
    Returning the number of logical cores instead. You can silence this warning by setting LOKY_MAX_CPU_COUNT to the number of cores you want to use.
      warnings.warn(
      File "e:\UpdatedProject Venv on Python 3.12.2\.venv\Lib\site-packages\joblib\externals\loky\backend\context.py", line 282, in _count_physical_cores
        raise ValueError(f"found {cpu_count_physical} physical cores < 1")
    100%|██████████| 166/166 [05:25<00:00,  1.96s/it]
    100%|██████████| 12/12 [00:21<00:00,  1.81s/it]
:::

::: {.output .stream .stdout}

    ------------------------------------
    Наилучшая метрика 27.947
    Оптимальное число лагов в признаках 474
    Оптимальная ширина окна 12
:::

::: {.output .stream .stderr}
:::

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/76bad69928cf3f7649f5a4813c50d2708b0a2158.png)
:::

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/06ba73a026999ef8e7f12bfb93f9b109461200c5.png)
:::
:::

::: {.cell .markdown}
##### Подбор гиперпараметров модели при оптимальных показателях генерации признаков {#подбор-гиперпараметров-модели-при-оптимальных-показателях-генерации-признаков}

-   ключевой кол-во деревьев (`n_estimators`)
-   l1 и l2 регуляризацию (`reg_alpha`, `reg_lambda`)
:::

::: {.cell .code execution_count="34"}
``` python
# После изысканий выше знаем оптимальное число лагов и величины окон.
# Сделаем из них датасет для подбора гиперпараметров и замера временени обучения

data = df_train_valid.copy()
func_make_features(data, max_lag=best_LGBM_lag, rolling_size=best_LGBM_roll)

x = data.drop(columns='num_orders')
y = data['num_orders']
```
:::

::: {.cell .code execution_count="35"}
``` python
model = LGBMRegressor(random_state=RANDOM_STATE, device='GPU', verbose=-1)
params = {'n_estimators':[100, 200],             # default 100
          'reg_alpha':[0., 0.3, 0.6, 0.9, 1.0],  # default 0. L1 regularization term on weights
          'reg_lambda':[0., 0.3, 0.6, 0.9, 1.0]  # default 0. L2 regularization term on weights
          }

grid_LGBM = GridSearchCV(estimator=model,
                            param_grid=params,
                            cv=TimeSeriesSplit(n_splits=3).split(x),
                            scoring='neg_mean_squared_error')

grid_LGBM.fit(x, y)
```

::: {.output .execute_result execution_count="35"}
```{=html}
<style>#sk-container-id-3 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-3 {
  color: var(--sklearn-color-text);
}

#sk-container-id-3 pre {
  padding: 0;
}

#sk-container-id-3 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-3 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-3 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-3 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-3 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-3 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-3 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-3 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-3 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-3 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-3 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-3 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-3 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-3 div.sk-label label.sk-toggleable__label,
#sk-container-id-3 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-3 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-3 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-3 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-3 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-3 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-3 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-3 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-3 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=&lt;generator object TimeSeriesSplit.split at 0x000002879EF897A0&gt;,
             estimator=LGBMRegressor(device=&#x27;GPU&#x27;, random_state=12345,
                                     verbose=-1),
             param_grid={&#x27;n_estimators&#x27;: [100, 200],
                         &#x27;reg_alpha&#x27;: [0.0, 0.3, 0.6, 0.9, 1.0],
                         &#x27;reg_lambda&#x27;: [0.0, 0.3, 0.6, 0.9, 1.0]},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=&lt;generator object TimeSeriesSplit.split at 0x000002879EF897A0&gt;,
             estimator=LGBMRegressor(device=&#x27;GPU&#x27;, random_state=12345,
                                     verbose=-1),
             param_grid={&#x27;n_estimators&#x27;: [100, 200],
                         &#x27;reg_alpha&#x27;: [0.0, 0.3, 0.6, 0.9, 1.0],
                         &#x27;reg_lambda&#x27;: [0.0, 0.3, 0.6, 0.9, 1.0]},
             scoring=&#x27;neg_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">estimator: LGBMRegressor</label><div class="sk-toggleable__content fitted"><pre>LGBMRegressor(device=&#x27;GPU&#x27;, random_state=12345, verbose=-1)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">LGBMRegressor</label><div class="sk-toggleable__content fitted"><pre>LGBMRegressor(device=&#x27;GPU&#x27;, random_state=12345, verbose=-1)</pre></div> </div></div></div></div></div></div></div></div></div>
```
:::
:::

::: {.cell .code execution_count="36"}
``` python
print('Оптимальные гиперпараметры модели', grid_LGBM.best_params_)
print('Наилучшая метрика:', round((grid_LGBM.best_score_ * -1) ** 0.5, 3))

# Сохраним итоговый результат для сводной таблицы 
best_LGBM_rmse = (grid_LGBM.best_score_ * -1) ** 0.5
```

::: {.output .stream .stdout}
    Оптимальные гиперпараметры модели {'n_estimators': 200, 'reg_alpha': 1.0, 'reg_lambda': 0.0}
    Наилучшая метрика: 23.478
:::
:::

::: {.cell .markdown}
##### Время обучения {#время-обучения}

Обучим модель с подобранными гипераметрами на обучающей и валидационной
выборках.\
Засечём время обучения.
:::

::: {.cell .code execution_count="37"}
``` python
LGBM_model_best = LGBMRegressor(random_state=RANDOM_STATE, **grid_LGBM.best_params_, verbose=-1) 

start = time()
LGBM_model_best.fit(x, y)
end = time()

LGBM_time = round((end - start), 2)

print('_______________')
print('Время обучения модели', LGBM_time)
```

::: {.output .stream .stdout}
    _______________
    Время обучения модели 1.4
:::
:::

::: {.cell .markdown}
### Prophet

Специализированная модель для прогнозирования временных рядов.\
Попробуем её \"из коробки\" и с перебором гиперпараметров на
кросс-валидации.
:::

::: {.cell .markdown}
#### Подготовка дата-сета
:::

::: {.cell .code execution_count="38"}
``` python
data_prophet = df_train_valid.copy()

# Убираем 20% на валидацию и задаём горизонт прогнозирования
predictions = len(df_train_valid) * 0.2

# приводим dataframe к нужному формату
data_prophet  = data_prophet.reset_index()
data_prophet.columns = ['ds', 'y']

# разбиваем на обучающую и валидационную выборки

prophet_train, prophet_valid = train_test_split(data_prophet, test_size=0.2, shuffle=False)
```
:::

::: {.cell .markdown}
#### Прогнозирование \"из коробки\"
:::

::: {.cell .code execution_count="39"}
``` python
prophet_model = Prophet()
prophet_model.fit(prophet_train);
```
:::

::: {.cell .code execution_count="40"}
``` python
# Зададим интервал прогнозирования
future = prophet_model.make_future_dataframe(periods=len(prophet_valid), freq='h')

forecast = prophet_model.predict(future)

print('Некоторые признаки, генерируемые моделью Prophet \n')
forecast[['ds', 'yhat', 'yhat_lower',
		'yhat_upper', 'trend',
		'trend_lower', 'trend_upper']].tail()
```

::: {.output .stream .stdout}
    Некоторые признаки, генерируемые моделью Prophet 
:::

::: {.output .execute_result execution_count="40"}
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
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3969</th>
      <td>2018-08-13 09:00:00</td>
      <td>118.034284</td>
      <td>87.854874</td>
      <td>144.342511</td>
      <td>106.598163</td>
      <td>105.762930</td>
      <td>107.414307</td>
    </tr>
    <tr>
      <th>3970</th>
      <td>2018-08-13 10:00:00</td>
      <td>122.893532</td>
      <td>95.103614</td>
      <td>154.618243</td>
      <td>106.614024</td>
      <td>105.776763</td>
      <td>107.431652</td>
    </tr>
    <tr>
      <th>3971</th>
      <td>2018-08-13 11:00:00</td>
      <td>111.924414</td>
      <td>83.925932</td>
      <td>139.892372</td>
      <td>106.629885</td>
      <td>105.790929</td>
      <td>107.448996</td>
    </tr>
    <tr>
      <th>3972</th>
      <td>2018-08-13 12:00:00</td>
      <td>98.392101</td>
      <td>69.482884</td>
      <td>127.249463</td>
      <td>106.645746</td>
      <td>105.804812</td>
      <td>107.466340</td>
    </tr>
    <tr>
      <th>3973</th>
      <td>2018-08-13 13:00:00</td>
      <td>96.044529</td>
      <td>65.952070</td>
      <td>124.758978</td>
      <td>106.661607</td>
      <td>105.818695</td>
      <td>107.483685</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
Модель выдаём данные факт + прогноз в одном датасете и множество данных,
в т.ч. интервалы по прогнозу и тренду.\
Нам для вычисления метрики нужен только пронгоз (признак \'yhat\').
:::

::: {.cell .code execution_count="41"}
``` python
# Модель делает единый датасет факт + прогноз. Возьмём только прогноз.
preds = forecast[['ds', 'yhat']]
preds = preds.set_index('ds').tail(len(prophet_valid))
```
:::

::: {.cell .code execution_count="42"}
``` python
rmse = mean_squared_error(preds['yhat'], prophet_valid['y'])**0.5
print('Наилучшая метрика "из коробки":', round(rmse, 3))
```

::: {.output .stream .stdout}
    Наилучшая метрика "из коробки": 32.203
:::
:::

::: {.cell .markdown}
Результат не блестящий для модели специально разработанной для задач
временных рядов.\
Может быть настройка гиперпараметров поможет, модель Prophet аддитивной
регрессией; обычно регрессии чутко реагируют на гиперпараметры.
:::

::: {.cell .markdown}
#### Подбор гиперпараметров на кросс-валидации

У модели свой встроенный метод кросс-валидации, используем его.
Оптимизируем метрику через подбор гиперпараметров на кросс-валидации:

-   l1/l2 регуляризация точек изменения (`changepoint_prior_scale`)
-   подход к оценке сезонности (`seasonality_mode`)
-   l1/l2 регуляризация сезонности (`changepoint_prior_scale`)
-   т.к. в данных есть и недельная и внутрисуточная сезонность явно их
    отметим (`weekly_seasonality` и `daily_seasonality`)
:::

::: {.cell .code execution_count="43"}
``` python
param_grid = {'changepoint_prior_scale':[0.001, 0.05, 0.08, 0.5],  # default 0.05 нет регуляризации (L1-L2 регуляризация)
              'seasonality_mode':['additive', 'multiplicative'],   # default 'additive'  
              'seasonality_prior_scale':[0.01, 1, 5, 10, 12],      # default 10 нет регуляризации (L1-L2 регуляризация)
              'weekly_seasonality':[True],                         # default auto, в данных явно есть недельная сезонность
              'daily_seasonality':[True]                           # default auto, в данных явно есть суточная сезонность          
             }

# Комбинации возможных параметров
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

#Список для хранения метрики для каждой комбинации
rmses =[]

for params in all_params:
    model = Prophet(**params).fit(data_prophet)
    df_cv = cross_validation(model, horizon='96 hours', parallel='processes') # на параметр horizon завязано число фолдов
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses

# Лучшая метрика и оптимизированные гиперпараметры
best_Prophet_score = tuning_results['rmse'].min()
best_Prophet_params = all_params[np.argmin(rmses)]

print('Оптимальные гиперпараметры модели', best_Prophet_params)
print('Наилучшая метрика:', round(best_Prophet_score, 3))
```

::: {.output .stream .stdout}
    Оптимальные гиперпараметры модели {'changepoint_prior_scale': 0.05, 'seasonality_mode': 'additive', 'seasonality_prior_scale': 10, 'weekly_seasonality': True, 'daily_seasonality': True}
    Наилучшая метрика: 25.187
:::
:::

::: {.cell .markdown}
##### Время обучения {#время-обучения}

Обучим модель с подобранными гипераметрами на обучающей и валидационной
выборках.\
Засечём время обучения.
:::

::: {.cell .code execution_count="44"}
``` python
Prophet_model_best = Prophet(**best_Prophet_params)

start = time()
Prophet_model_best.fit(data_prophet)
end = time()

Prophet_time = round((end - start), 2)

print('_______________')
print('Время обучения модели', Prophet_time)
```

::: {.output .stream .stdout}
    _______________
    Время обучения модели 0.33
:::
:::

::: {.cell .markdown}
### Сводные данные по итогам кросс-валидации моделей

**Выводы:**\
1) Наилучшие результаты по метрике и скорости обучения показала модель
Elastic Net.

2\) Близкая метрика у бустинга LightGBM при более длительном(в 10 раз
дольше) обучении.

3\) Интересно, что специализированная модель Prophet показала результаты
чуть хуже, чем универсальные модели + настраиваемая генерация
синтетических признаков.

4\) Оптимальное кол-во лагов и ширина скользящих окон различаются в
зависимости от типа модели\
(даже у \"родственных\" Линейной регрессии и Elastic Net)
:::

::: {.cell .code execution_count="45"}
``` python
score_table = [[round(best_LGBM_rmse, 3), LGBM_time, best_LGBM_lag, best_LGBM_roll],
               [round(best_Elastic_rmse, 3), Elastic_time, best_Elastic_lag, best_Elastic_roll],
                [round(best_LR_rmse, 3), LR_time, best_LR_lag, best_LR_roll],
                [round(best_Prophet_score, 3), Prophet_time, 'n.a.', 'n.a.'] 
                ]

pd.DataFrame(index=['LIGHT_GBM', 'ELASTIC NET', 'LINEAR_REGRESSION', 'PROPHET'], 
             data = score_table, columns=['МЕТРИКА RMSE НА КРОСС-ВАЛИДАЦИИ', 'ВРЕМЯ ОБУЧЕНИЯ, С', 'КОЛ-ВО ЛАГОВ', 'ШИРИНА СКОЛЬЗЯЩИХ ОКОН']
             ).sort_values(by='МЕТРИКА RMSE НА КРОСС-ВАЛИДАЦИИ')
```

::: {.output .execute_result execution_count="45"}
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
      <th>МЕТРИКА RMSE НА КРОСС-ВАЛИДАЦИИ</th>
      <th>ВРЕМЯ ОБУЧЕНИЯ, С</th>
      <th>КОЛ-ВО ЛАГОВ</th>
      <th>ШИРИНА СКОЛЬЗЯЩИХ ОКОН</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ELASTIC NET</th>
      <td>23.188</td>
      <td>0.10</td>
      <td>336</td>
      <td>48</td>
    </tr>
    <tr>
      <th>LIGHT_GBM</th>
      <td>23.478</td>
      <td>1.40</td>
      <td>474</td>
      <td>12</td>
    </tr>
    <tr>
      <th>LINEAR_REGRESSION</th>
      <td>24.641</td>
      <td>0.10</td>
      <td>336</td>
      <td>12</td>
    </tr>
    <tr>
      <th>PROPHET</th>
      <td>25.187</td>
      <td>0.33</td>
      <td>n.a.</td>
      <td>n.a.</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
## Тестирование

Точность прогнозирования должна быть не выше 48 по метрике RMSE на
тестовой выборке в 10%.

В данном проекте:

-   мало тестовых данных (18 дней или \>500 наблюдений)
-   возможно, недостаточно (полгода или ок. 4 тыс. наблюдений) данных
    для качественного обучения моделей
-   есть ускорение темпов роста в конце периода, т.е. в тестовых данных

В таких условиях качество прогноза на тестовых данных будет сильно хуже,
чем на кросс-валидации.\
Поэтому отступим от классического подхода и проверим все модели.

Выберем модель для продакшн исходя из итоговой оценки на тестовых
данных.
:::

::: {.cell .code execution_count="46"}
``` python
# Для удобства будем вести сводную таблицу результатов
score_table_dict = {
  'test_rmse':[''],
  'cv_rmse':[round(best_LGBM_rmse, 3), round(best_Elastic_rmse, 3), round(best_LR_rmse, 3), round(best_Prophet_score, 3), np.nan],
  'fit_time':[LGBM_time, Elastic_time, LR_time, Prophet_time, np.nan]
} 

final_score_table = pd.DataFrame(score_table_dict, index=['LIGHT_GBM', 'ELASTIC_NET', 'LINEAR_REGRESSION', 'PROPHET', 'DUMMY_MEAN'])
```
:::

::: {.cell .markdown}
### Дамми-модель. {#дамми-модель}

Бейз-лайн для проверки на адекватность.

Сравним показатели дамми модели (предсказывает среднее значение) и наших
моделей на тестовых данных.
:::

::: {.cell .code execution_count="47"}
``` python
dummy_model = DummyRegressor()

data_dummy = df.copy()
data_dummy.sort_index(ascending=True, inplace=True)

func_make_features(data_dummy, max_lag=300, rolling_size=12)

dummy_train, dummy_test = train_test_split(data_dummy, test_size=0.1, shuffle=False)

x_dummy_train = dummy_train.drop('num_orders', axis=1) 
y_dummy_train = dummy_train['num_orders']

x_dummy_test = dummy_test.drop('num_orders', axis=1) 
y_dummy_test = dummy_test['num_orders']

dummy_model.fit(x_dummy_train, y_dummy_train)

preds = dummy_model.predict(x_dummy_test)

dummy_test_score = round(mean_squared_error(y_true=y_dummy_test, y_pred=preds)**0.5, 3)

print('Метрика дамми-модели:', dummy_test_score)
```

::: {.output .stream .stdout}
    Метрика дамми-модели: 84.752
:::
:::

::: {.cell .code execution_count="48"}
``` python
final_score_table.loc['DUMMY_MEAN', 'test_rmse'] = dummy_test_score
```
:::

::: {.cell .markdown}
### Elastic Net
:::

::: {.cell .code execution_count="49"}
``` python
# Сгенерируем синтетические признаки оптимальные для модели. 

test_data = df_test.copy()
func_make_features(data=test_data, max_lag=best_Elastic_lag, rolling_size=best_Elastic_roll)

x_test = test_data.drop('num_orders', axis=1)
y_test = test_data['num_orders']

# Метрика на тестовой выборке

preds = Elastic_model_best.predict(x_test)

Elastic_test_score = round(mean_squared_error(preds, y_test)**0.5, 3)

print('Метрика на тестовой выборке:', Elastic_test_score)
```

::: {.output .stream .stdout}
    Метрика на тестовой выборке: 59.24
:::
:::

::: {.cell .code execution_count="50"}
``` python
final_score_table.loc['ELASTIC_NET', 'test_rmse'] = Elastic_test_score
```
:::

::: {.cell .markdown}
### LightGBM {#lightgbm}
:::

::: {.cell .code execution_count="51"}
``` python
# Сгенерируем синтетические признаки оптимальные для модели. 

test_data = df_test.copy()
func_make_features(data=test_data, max_lag=best_LGBM_lag, rolling_size=best_LGBM_roll)

x_test = test_data.drop('num_orders', axis=1)
y_test = test_data['num_orders']

# Метрика на тестовой выборке

preds = LGBM_model_best.predict(x_test)

LGBM_test_score = round(mean_squared_error(preds, y_test)**0.5, 3)

print('Метрика на тестовой выборке:', LGBM_test_score)
```

::: {.output .stream .stdout}
    Метрика на тестовой выборке: 54.888
:::
:::

::: {.cell .code execution_count="52"}
``` python
final_score_table.loc['LIGHT_GBM', 'test_rmse'] = LGBM_test_score 
```
:::

::: {.cell .markdown}
### Линейная регрессия {#линейная-регрессия}
:::

::: {.cell .code execution_count="53"}
``` python
# Сгенерируем синтетические признаки оптимальные для модели. 

test_data = df_test.copy()
func_make_features(data=test_data, max_lag=best_LR_lag, rolling_size=best_LR_roll)

x_test = test_data.drop('num_orders', axis=1)
y_test = test_data['num_orders']

# Метрика на тестовой выборке

preds_lr = LR_model_best.predict(x_test)

LR_test_score = round(mean_squared_error(preds_lr, y_test)**0.5, 3)

print('Метрика на тестовой выборке:', LR_test_score)
```

::: {.output .stream .stdout}
    Метрика на тестовой выборке: 43.331
:::
:::

::: {.cell .code execution_count="54"}
``` python
final_score_table.loc['LINEAR_REGRESSION', 'test_rmse'] = LR_test_score
```
:::

::: {.cell .markdown}
### Prophet {#prophet}
:::

::: {.cell .code execution_count="55"}
``` python
data_prophet_full = df.copy()

# 10% на тестирование по условию
predictions = len(data_prophet_full) * 0.1

# приводим dataframe к нужному формату
data_prophet_full = data_prophet_full.reset_index()
data_prophet_full.columns = ['ds', 'y']

# разбиваем на обучающую и валидационную выборки
prophet_train_valid, prophet_test = train_test_split(data_prophet_full, test_size=0.1, shuffle=False)

# зададим интервал прогнозирования
future = Prophet_model_best.make_future_dataframe(periods=len(prophet_test), freq='h')

forecast = Prophet_model_best.predict(future)
```
:::

::: {.cell .markdown}
Модель выдаём данные факт + прогноз в одном датасете и множество данных,
в т.ч. интервалы по прогнозу и тренду.\
Нам для вычисления метрики нужен только пронгоз (признак \'yhat\').
:::

::: {.cell .code execution_count="56"}
``` python
# Модель делает единый датасет факт + прогноз. Возьмём только прогноз.
preds = forecast[['ds', 'yhat']]
preds = preds.set_index('ds').tail(len(prophet_test))
```
:::

::: {.cell .code execution_count="57"}
``` python
Prophet_test_score = round(mean_squared_error(preds['yhat'], prophet_test['y'])**0.5, 3)
print('Метрика на тестовой выборке:', round(rmse, 3))
```

::: {.output .stream .stdout}
    Метрика на тестовой выборке: 32.203
:::
:::

::: {.cell .code execution_count="58"}
``` python
final_score_table.loc['PROPHET', 'test_rmse'] = Prophet_test_score
```
:::

::: {.cell .markdown}
### Сводные данные по итогам тестирования моделей
:::

::: {.cell .code execution_count="59"}
``` python
final_score_table_exp = final_score_table.copy()
final_score_table_exp.columns = ['test_rmse', 'cv_rmse', 'fit_time']

final_score_table_exp['ratio'] = round(final_score_table_exp['test_rmse'].astype(float) / final_score_table_exp['cv_rmse'].astype(float), 2)
```
:::

::: {.cell .code execution_count="60"}
``` python
final_score_table['ОТНОШЕНИЕ МЕТРИК ТЕСТ/ВАЛ'] = round(final_score_table['test_rmse'].astype(float) / final_score_table['cv_rmse'].astype(float), 2)
final_score_table = final_score_table.rename(columns={'test_rmse':'ИТОГОВАЯ МЕТРИКА RMSE НА ТЕСТЕ', 'cv_rmse':'МЕТРИКА RMSE НА КРОСС-ВАЛИДАЦИИ', 
                                                      'fit_time':'ВРЕМЯ ОБУЧЕНИЯ, С'})
final_score_table.sort_values(by='ИТОГОВАЯ МЕТРИКА RMSE НА ТЕСТЕ')
```

::: {.output .execute_result execution_count="60"}
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
      <th>ИТОГОВАЯ МЕТРИКА RMSE НА ТЕСТЕ</th>
      <th>МЕТРИКА RMSE НА КРОСС-ВАЛИДАЦИИ</th>
      <th>ВРЕМЯ ОБУЧЕНИЯ, С</th>
      <th>ОТНОШЕНИЕ МЕТРИК ТЕСТ/ВАЛ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LINEAR_REGRESSION</th>
      <td>43.331</td>
      <td>24.641</td>
      <td>0.10</td>
      <td>1.76</td>
    </tr>
    <tr>
      <th>PROPHET</th>
      <td>48.243</td>
      <td>25.187</td>
      <td>0.33</td>
      <td>1.92</td>
    </tr>
    <tr>
      <th>LIGHT_GBM</th>
      <td>54.888</td>
      <td>23.478</td>
      <td>1.40</td>
      <td>2.34</td>
    </tr>
    <tr>
      <th>ELASTIC_NET</th>
      <td>59.24</td>
      <td>23.188</td>
      <td>0.10</td>
      <td>2.55</td>
    </tr>
    <tr>
      <th>DUMMY_MEAN</th>
      <td>84.752</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
-   **Цель по точности на тестовых данных достигает только Линейная
    регрессия**\
-   Prophet чуть-чуть не дотягивает. Остальные модели существенно хуже
-   Все модели адекватны и прогнозируют существенно лучше дамми-модели
    по среднему
:::

::: {.cell .markdown}
### Визуализация прогноза и факта на тестовых данных модели-лидера
:::

::: {.cell .code execution_count="61"}
``` python
df_results = pd.DataFrame(columns=['forecast','fact'], index=y_test.index)
df_results['forecast'] = preds_lr
df_results['fact'] = y_test
```
:::

::: {.cell .code execution_count="62"}
``` python
df_results.plot(figsize=(15,5), legend=True, title='Сравнение прогноза и факта Линейной регрессии');
```

::: {.output .display_data}
![](https://github.com/sotwra/Portfolio/upload/main/Taxi%20orders%20forecasting/Forecasting_Taxi_Picks/b817c98d6be426c02aca7cdf29e61976876f3dba.png)
:::
:::

::: {.cell .markdown}
Модель неплохо предсказывает динамику данных, но не справляется со
выбросами.\
В некотором роде модель \"стесняется\" предсказывать очень большие
значения.
:::

::: {.cell .markdown}
## Общие выводы по проекту

1\) Лучшие результаты по точности и скорости показала Линейная
регрессия.\
Можно использовать эту модель в продакшн.

-   Линейная регрессия единственная достигла требуемый заказчиком
    уровень точности.
-   Модель неплохо предсказывает динамику данных, но не справляется с
    большими всплесками.
-   В проекте критически важно не переобучить модель. Так,
    \"родственная\" Elastic Net с регуляризаций плохо справляется с
    тестовыми данными, хотя точнее всех работает на кросс-валидации.

2\) Проект работает с нестационарными данными, т.к. имеются:

-   Возрастающий тренд, причём с ускорением темпов в ок. середины
    рассматриваемого периода.
-   Сезонность как внутри суток, так и недельная.
-   Заметное количеством выбросов до 6 раз превышающих медианное
    значение.

3\) Данных мало (6 мес.). Этого может быть не достаточно для раскрытия
потенциала рассмотренных моделей, а так же использования нейросетей.\
Это подтверждается тем, что при тестировании результаты у рассмотренных
моделей до 2,5 раза хуже чем на валидации.

4\) Модели общего назначения сложнее в использовании, чем
специализированная Prophet:

-   Они требуют генерации отдельного, зависящего от типа модели набора
    признаков. Это потенциально может повлиять на надёжность решения.
-   Prophet автоматически генерирует нужный набор признаков + имеет
    хорошо интерпретируемые гиперпараметры.

**Возможные пути улучшения решения**

1\) Запросить у заказчика данные за более длительный период (2-5 лет) и
обучить и протестировать на нём модели 2) Изучить возможности применения
в проекте:

-   Стекинга моделей, например использования сочетания Prophet + Light
    GBM
-   Предобученных и собственных нейросетей
-   Приведения данных к стационарному виду и использования
    специализированных моделей семейства ARIMA
-   Стандартизированных синтетических признаков, например, с помощью
    библиотеки Tsfresh
:::
