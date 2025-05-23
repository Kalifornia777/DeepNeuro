# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:30:32 2025

@author: AM4
"""

# Серегин К.С. 23ВП2 Задание 2, Лабораторная работа 2

# Попробуем обучить один нейрон на задачу классификации двух классов

import pandas as pd  # библиотека pandas нужна для работы с данными
import matplotlib.pyplot as plt  # matplotlib для построения графиков
import numpy as np  # numpy для работы с векторами и матрицами

# Считываем данные 
# df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#     'machine-learning-databases/iris/iris.data', header=None)

df = pd.read_csv('data.csv')

# смотрим что в них
print(df.head())

# три столбца - это признаки, четвертый - целевая переменная (то, что мы хотим предсказывать)

# выделим целевую переменную в отдельную переменную
y = df.iloc[:, 4].values

# так как ответы у нас строки - нужно перейти к численным значениям
y = np.where(y == "Iris-setosa", 1, -1)

# возьмем три признака
X = df.iloc[:, [0, 1, 2]].values  # Теперь 3 признака

# Признаки в X, ответы в y - посмотрим на плоскости как выглядит задача
# Визуализируем только два признака для наглядности

###
plt.figure()
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', marker='o', label='Класс 1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', marker='x', label='Класс -1')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.title('Визуализация данных (первые 2 признака)')
plt.show()
###

# переходим к созданию нейрона
# функция нейрона:
# значение = w1*признак1 + w2*признак2 + w3*признак3 + w0
# ответ = 1, если значение >= 0
# ответ = -1, если значение < 0

###
def neuron(w, x):
    value = w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[0]  # Учитываем три признака
    if value >= 0:
        predict = 1
    else:
        predict = -1
    return predict
###

###
# проверим как это работает (веса зададим пока произвольно)
w = np.array([0, 0.1, 0.4, 0.2])  # Теперь четыре веса: w0, w1, w2, w3
print("Предсказание для первого примера:", neuron(w, X[1]))  # вывод ответа нейрона для примера с номером 1

# теперь создадим процедуру обучения
# корректировка веса производится по выражению:
# w_new = w_old + eta * x * (target - predict)

# зададим начальные значения весов
w = np.random.random(4)  # теперь четыре веса: w0, w1, w2, w3
eta = 0.01  # скорость обучения
w_iter = []  # пустой список, в него будем добавлять веса, чтобы потом построить график

for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w, xi)
    w[1:] += eta * (target - predict) * xi  # корректировка весов для признаков
    w[0] += eta * (target - predict)  # корректировка смещения
    # каждую 10ю итерацию будем сохранять набор весов в специальном списке
    if j % 10 == 0:
        w_iter.append(w.tolist())

# посчитаем ошибки
sum_err = 0
for xi, target in zip(X, y):
    predict = neuron(w, xi)
    sum_err += (target - predict) / 2

print("Всего ошибок:", sum_err)

# визуализация процесса обучения (для первых двух признаков)
xl = np.linspace(min(X[:, 0]), max(X[:, 0]))  # диапазон координаты x для построения линии

# построим сначала данные на плоскости
plt.figure()
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', marker='o', label='Класс 1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', marker='x', label='Класс -1')

# потом в цикле будем брать набор весов из сохраненного списка и по нему строить линию
for i, w in zip(range(len(w_iter)), w_iter):
    yl = -(xl * w[1] + w[0]) / w[2]  # уравнение линии для первых двух признаков
    plt.plot(xl, yl, label=f'Итерация {i}')  # строим разделяющую границу
    plt.pause(1)

plt.text(xl[-1] - 0.3, yl[-1], 'END', dict(size=14, color='red'))
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.title('Процесс обучения (первые два признака)')
plt.show()