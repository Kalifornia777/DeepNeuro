from random import randint

# Серегин К.С. 23ВП2 Задание 1, Лабораторная работа 2

# 1 создайте список, заполненный случайными числами
random_numbers = [randint(10, 100) for _ in range(100)]

print("Список случайных чисел:", random_numbers)

Esum = 0  

# 2 создайте цикл, который проходит все элементы списка и суммирует только четные значения
for number in random_numbers:
    if number % 2 == 0:  
        Esum += number  
# 3 выведите полученную сумму на экран в консоли
print("Полученная сумма:", Esum)