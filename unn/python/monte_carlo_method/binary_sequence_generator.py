import string
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def generate_sequence(N, p):
    """Генерирует случайную последовательность нулей и единиц."""
    return ''.join(np.random.choice(['0', '1'], size=N, p=[p, 1 - p]))


def count_substring(sequence, substring):
    """Считает количество вхождений подстроки в последовательность."""
    return sum(1 for i in range(len(sequence) - len(substring) + 1) if sequence[i:i + len(substring)] == substring)


def distribution_analysis(sequence, substring, num_trials=1000):
    """Анализирует распределение числа вхождений подстроки."""
    counts = [count_substring(generate_sequence(len(sequence), p), substring) for _ in range(num_trials)]
    mean = np.mean(counts)
    variance = np.var(counts)

    # Построение гистограммы
    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=range(min(counts), max(counts) + 2), alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Число вхождений')
    plt.ylabel('Частота')
    plt.title(f'Распределение числа вхождений "{substring}" (Среднее={mean:.2f}, Дисперсия={variance:.2f})')
    plt.grid(True)
    plt.show()

    return mean, variance


# Пример использования
N = input('Введите длину последовательности: ')  # Длина последовательности
N = int(N)
p = input('Введите вероятность единицы: ')  # Вероятность единицы
p = float(p)
substring = input('Введите искомую подстроку: ')  # Искомая строка в исходной последовательности

sequence = generate_sequence(N, p)
count = count_substring(sequence, substring)
mean, variance = distribution_analysis(sequence, substring)

print(f'Последовательность: {sequence[:50]}...')
print(f'Число вхождений "{substring}": {count}')
print(f'Выборочное среднее: {mean:.2f}')
print(f'Выборочная дисперсия: {variance:.2f}')