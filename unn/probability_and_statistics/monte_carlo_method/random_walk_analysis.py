import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

def random_walk_first_hitting_time(a, b, T, N):
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

def plot_hitting_time_cdf(hitting_times):
    """
    Строит эмпирическую функцию распределения (CDF) и накладывает нормальное и логнормальное распределения.

    Args:
        hitting_times (list): Список моментов первого достижения уровня T.
    """

    sorted_hitting_times = np.sort(hitting_times)
    cdf = np.arange(1, len(sorted_hitting_times) + 1) / len(sorted_hitting_times)

    # Оценка параметров нормального распределения
    mean_est = np.mean(hitting_times)
    std_est = np.std(hitting_times)

    # Генерация значений нормального распределения
    norm_cdf = norm.cdf(sorted_hitting_times, mean_est, std_est)

    # Оценка параметров логнормального распределения
    shape, loc, scale = lognorm.fit(hitting_times, floc=0)

    # Генерация значений логнормального распределения
    lognorm_cdf = lognorm.cdf(sorted_hitting_times, shape, loc=loc, scale=scale)

    # Построение графика
    plt.plot(sorted_hitting_times, cdf, label='Эмпирическая CDF')
    plt.plot(sorted_hitting_times, norm_cdf, label='Нормальное распределение', linestyle='--')
    plt.plot(sorted_hitting_times, lognorm_cdf, label='Логнормальное распределение', linestyle=':')
    plt.xlabel('Момент первого достижения уровня T')
    plt.ylabel('CDF')
    plt.title('Эмпирическая CDF и распределения')
    plt.legend()
    plt.show()

# Запрашиваем параметры у пользователя
a = float(input("Введите левую границу равномерного распределения (a): "))
b = float(input("Введите правую границу равномерного распределения (b): "))
T = float(input("Введите уровень достижения (T): "))
N = int(input("Введите объем выборки (N): "))

# Выполняем моделирование
hitting_times, mean_hitting_time, variance_hitting_time = random_walk_first_hitting_time(a, b, T, N)

# Выводим результаты
print(f"Выборочное среднее: {mean_hitting_time}")
print(f"Выборочная дисперсия: {variance_hitting_time}")

# Строим график
plot_hitting_time_cdf(hitting_times)