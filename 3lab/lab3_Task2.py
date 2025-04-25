# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 12:09:02 2025

@author: User
"""

# Серегин К.С. 23ВП2 — Лабораторная работа 3, Задание 2

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загружаем данные
data = load_iris()
features = data.data
labels = data.target

# Делим на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Стандартизация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Переводим в тензоры
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

# Определяем модель
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 10)
        self.output = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)

model = NeuralNet()

# Функция потерь и оптимизатор
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Тренировка модели
for epoch in range(10000):
    predictions = model(X_train)
    loss = loss_fn(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Эпоха [{epoch + 1}/10000], Потери: {loss.item():.4f}")

# Оценка точности на тестовой выборке
with torch.no_grad():
    test_preds = model(X_test)
    predicted_classes = torch.argmax(test_preds, dim=1)
    accuracy = (predicted_classes == y_test).float().mean()
    print(f"Точность на тестовой выборке: {accuracy.item() * 100:.2f}%")
