import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Серегин К.С. 23ВП2 — Лабораторная работа 4, Задание 1, вариант 26

# 1. Загрузка датасета
data = pd.read_csv('dataset_simple.csv')

# 2. Формирование матрицы признаков и целевой переменной
ages = data[['age']].values
X = np.c_[np.ones(ages.shape[0]), ages]  # Добавление единичного столбца (intercept)
y = data['income'].to_numpy()

# 3. Вычисление коэффициентов линейной регрессии
XT_X = X.T @ X
XT_y = X.T @ y
weights = np.linalg.inv(XT_X) @ XT_y

# 4. Отображение коэффициентов
print("\nАналитическое решение (пасхалка линейной алгебры)")
print(f"intercept = {weights[0]:.2f}, slope = {weights[1]:.2f}")

# 5. Предсказания модели
y_pred = X @ weights
print("\nПримеры предсказаний:")
for idx, (age, actual, pred) in enumerate(zip(data['age'], y, y_pred)):
    if idx >= 5:
        break
    print(f"Возраст: {age} | Доход (факт): {actual} | Доход (прогноз): {pred:.2f}")

# 6. Построение графика
plt.figure(figsize=(10, 6))
plt.scatter(data['age'], y, label='Наблюдения', color='skyblue')
plt.plot(data['age'], y_pred, label='Модель', color='crimson', linewidth=2)
plt.title('Зависимость дохода от возраста')
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
