import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm

def generate_sequence(N, p):
    """Генерирует случайную последовательность нулей и единиц."""
    return ''.join(np.random.choice(['0', '1'], size=N, p=[p, 1 - p]))

def count_substring(sequence, substring):
    """Считает количество вхождений подстроки в последовательность."""
    return sum(1 for i in range(len(sequence) - len(substring) + 1) if sequence[i:i + len(substring)] == substring)

def distribution_analysis(sequence, substring, num_trials=1000):
    """Анализирует распределение числа вхождений подстроки и сравнивает с нормальным распределением."""
    counts = [count_substring(generate_sequence(len(sequence), p), substring) for _ in range(num_trials)]
    mean = np.mean(counts)
    variance = np.var(counts)
    std_dev = np.sqrt(variance)

    # Построение нормированной гистограммы
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins='auto', density=True, alpha=0.7, color='b', edgecolor='black', label='Нормированная гистограмма')
    plt.xlabel('Число вхождений')
    plt.ylabel('Плотность вероятности')
    plt.title(f'Распределение числа вхождений "{substring}" (Среднее={mean:.2f}, Дисперсия={variance:.2f})')
    plt.grid(True)

    # Построение плотности нормального распределения
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p_norm = norm.pdf(x, mean, std_dev)
    plt.plot(x, p_norm, 'r--', linewidth=2, label=f'Нормальное распределение (μ={mean:.2f}, σ={std_dev:.2f})')

    plt.legend()
    plt.show()

    return mean, variance

# Пример использования
N_input = input('Введите длину последовательности: ')  # Длина последовательности
N = int(N_input)
p_input = input('Введите вероятность единицы (от 0 до 1): ')  # Вероятность единицы
p = float(p_input)
substring = input('Введите искомую подстроку: ')  # Искомая строка в исходной последовательности

sequence = generate_sequence(N, p)
count = count_substring(sequence, substring)
mean, variance = distribution_analysis(sequence, substring)

print(f'Сгенерированная последовательность (первые 50 символов): {sequence[:50]}...')
print(f'Число вхождений подстроки "{substring}": {count}')
print(f'Выборочное среднее числа вхождений (по {1000} испытаниям): {mean:.2f}')
print(f'Выборочная дисперсия числа вхождений (по {1000} испытаниям): {variance:.2f}')