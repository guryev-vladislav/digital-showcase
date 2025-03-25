import numpy as np
import matplotlib.pyplot as plt

def random_walk_first_hitting_time(a, b, T, N):
    """
    Вычисляет момент первого достижения уровня T для случайного блуждания.

    Args:
        a (float): Левая граница равномерного распределения.
        b (float): Правая граница равномерного распределения.
        T (float): Уровень достижения.
        N (int): Объем выборки.

    Returns:
        tuple: Кортеж, содержащий:
            - hitting_times (list): Список моментов первого достижения уровня T.
            - mean_hitting_time (float): Выборочное среднее моментов первого достижения.
            - variance_hitting_time (float): Выборочная дисперсия моментов первого достижения.
    """

    hitting_times = []
    for _ in range(N):
        X = 0
        n = 0
        while abs(X) < T:
            xi = np.random.uniform(a, b)
            X += xi
            n += 1
        hitting_times.append(n)

    mean_hitting_time = np.mean(hitting_times)
    variance_hitting_time = np.var(hitting_times)

    return hitting_times, mean_hitting_time, variance_hitting_time

def plot_hitting_time_distribution(hitting_times):
    """
    Строит выборочную функцию распределения для моментов первого достижения.

    Args:
        hitting_times (list): Список моментов первого достижения уровня T.
    """

    plt.hist(hitting_times, bins=20, density=True, alpha=0.6, color='g')
    plt.xlabel('Момент первого достижения уровня T')
    plt.ylabel('Вероятность')
    plt.title('Выборочная функция распределения моментов первого достижения')
    plt.show()

# Запрашиваем параметры у пользователя
a = float(input("Введите левую границу равномерного распределения (a): "))
b = float(input("Введите правую границу равномерного распределения (b): "))
T = float(input("Введите уровень достижения (T): "))
N = int(input("Введите объем выборки (N): "))
# Задаем параметры
a = -1
b = 1
T = 10
N = 1000

# Выполняем моделирование
hitting_times, mean_hitting_time, variance_hitting_time = random_walk_first_hitting_time(a, b, T, N)

# Выводим результаты
print(f"Выборочное среднее: {mean_hitting_time}")
print(f"Выборочная дисперсия: {variance_hitting_time}")

# Строим график
plot_hitting_time_distribution(hitting_times)