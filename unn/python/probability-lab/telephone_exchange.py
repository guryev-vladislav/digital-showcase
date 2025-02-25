import numpy as np
import matplotlib.pyplot as plt
import collections
import math
import random
from scipy import stats

def poisson_random_variable(lambd, t, num_simulations):
    """Генерирует случайные числа Пуассона."""
    results = []
    for _ in range(num_simulations):
        u = random.random()
        k = 0
        p = math.exp(-lambd * t)
        F = p
        while u > F:
            k += 1
            p *= (lambd * t) / k
            F += p
        results.append(k)
    return results

def create_frequency_table(data):
    """Создает таблицу частот."""
    if not data: # Проверка на пустой список
        return {"yi": [], "ni": [], "ni/n": []}
    counter = collections.Counter(data)
    n = len(data)
    sorted_items = sorted(counter.items())
    yi = [item[0] for item in sorted_items]
    ni = [item[1] for item in sorted_items]
    ni_n = [item[1] / n for item in sorted_items]
    return {"yi": yi, "ni": ni, "ni/n": ni_n}

def print_frequency_table(frequency_table):
    """Печатает таблицу частот."""
    if not frequency_table["yi"]:
        print("Нет данных для отображения.")
        return
    print("yi\tni\tni/n")
    for i in range(len(frequency_table["yi"])):
        print(f"{frequency_table['yi'][i]}\t{frequency_table['ni'][i]}\t{frequency_table['ni/n'][i]:.4f}")

def calculate_sample_variance(data):
    """Вычисляет выборочную дисперсию."""
    mean = np.mean(data)
    return np.sum((data - mean) ** 2) / len(data) # Используем numpy для эффективности

def calculate_sample_median(data):
    """Вычисляет выборочную медиану."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid_index = n // 2
    if n % 2 == 0:
        return (sorted_data[mid_index - 1] + sorted_data[mid_index]) / 2
    else:
        return sorted_data[mid_index]

def calculate_characteristics(results, lambd, t):
    """Вычисляет характеристики выборки."""
    E_eta = lambd * t
    x_bar = np.mean(results)
    S_squared = calculate_sample_variance(results)
    D_eta = E_eta
    median = calculate_sample_median(results)
    range_val = np.max(results) - np.min(results)
    return {
        "Eη": E_eta, "x": x_bar, "abs_diff_mean": abs(E_eta - x_bar),
        "Dη": D_eta, "S^2": S_squared, "abs_diff_variance": abs(D_eta - S_squared),
        "Me": median, "Rb": range_val
    }

def plot_cdf(results, lambd, t):
    """Строит графики CDF."""
    n = len(results)
    sorted_data = np.sort(results)
    empirical_cdf = np.arange(1, n + 1) / n
    x_values = np.arange(max(results) + 1)
    theoretical_cdf = np.cumsum(stats.poisson.pmf(x_values, lambd * t))
    plt.step(sorted_data, empirical_cdf - 0.2, label="Выборочная ФР (смещенная)") # Убрано лишнее слово "функция"
    plt.step(x_values, theoretical_cdf, label="Теоретическая ФР") # Убрано лишнее слово "функция"
    plt.xlabel("Число вызовов (η)")
    plt.ylabel("F(x)")
    plt.title("Функции распределения (выборочная и теоретическая)")
    plt.legend()
    plt.show()

def kolmogorov_smirnov_test(data, lambd, t):
    """Тест Колмогорова-Смирнова."""
    n = len(data)
    x_values = np.arange(max(data) + 1)
    theoretical_cdf = np.cumsum(stats.poisson.pmf(x_values, lambd * t))
    D = max(abs(np.interp(x_values, np.sort(data), np.arange(1, n + 1) / n) - theoretical_cdf))
    return D

def goodness_of_fit_table(frequency_table, lambd, t):
    """Таблица согласия."""
    yi = frequency_table['yi']
    ni = frequency_table['ni']
    n = sum(ni)
    theoretical_probs = [stats.poisson.pmf(y, lambd * t) for y in yi]
    ni_n = [i / n for i in ni]
    max_deviation = max([abs(a - b) for a, b in zip(ni_n, theoretical_probs)])
    return {
        "yj": yi, "P({η = yj})": theoretical_probs,
        "nj/n": ni_n, "Max Deviation": max_deviation
    }

def print_goodness_of_fit_table(table_data):
    """Печатает таблицу согласия."""
    print("yj\tP({η=yj})\tnj/n")
    for i in range(len(table_data['yj'])):
        print(f"{table_data['yj'][i]}\t{table_data['P({η = yj})'][i]:.4f}\t{table_data['nj/n'][i]:.4f}")
    print(f"Максимальное отклонение: {table_data['Max Deviation']:.4f}")

def get_interval_boundaries():
    """Запрашивает границы интервалов."""
    while True:
        try:
            k = int(input("Введите число интервалов k ( > 1): ")) # Уточнение в запросе
            if k <= 1: raise ValueError
            break
        except ValueError:
            print("Некорректный ввод. Введите целое число > 1.")

    boundaries = []
    if k > 2:
        print(f"Введите {k - 1} границ (через пробел, по возрастанию):")
        while True:
            try:
                boundaries = [float(b) for b in input().split()]
                if len(boundaries) != k - 1: raise ValueError
                for i in range(len(boundaries) - 1):
                    if boundaries[i] >= boundaries[i+1]: raise ValueError
                break
            except ValueError:
                print(f"Некорректный ввод. Введите {k-1} чисел по возрастанию.") # Уточнение ошибки
    elif k == 2:
        while True:
            try:
                boundaries = [float(input("Введите 1 границу z1:"))]
                break
            except ValueError:
                print("Некорректный ввод. Введите число.")
    return k, boundaries

def calculate_theoretical_interval_probabilities(lambd, t, k, boundaries):
    """Рассчитывает теоретические вероятности интервалов."""
    probs = []
    cdf = lambda x: stats.poisson.cdf(x, lambd * t) # Лямбда-функция для краткости

    if k == 2:
        z1 = boundaries[0]
        probs = [cdf(z1), 1 - cdf(z1)] # Интервалы (-inf, z1), [z1, +inf)
    elif k > 2:
        probs.append(cdf(boundaries[0])) # Интервал (-inf, z1)
        for i in range(k - 2):
            probs.append(cdf(boundaries[i+1]) - cdf(boundaries[i])) # Интервалы [zi, z(i+1))
        probs.append(1 - cdf(boundaries[-1])) # Интервал [z(k-1), +inf)
    else: raise ValueError("k должно быть > 1") # Для ясности, хотя k=1 уже исключено в get_interval_boundaries
    return probs

def calculate_observed_interval_frequencies(results, k, boundaries):
    """Рассчитывает наблюдаемые частоты интервалов."""
    freqs = [0] * k
    if k == 2:
        z1 = boundaries[0]
        for res in results:
            freqs[0] += (res < z1) # True/False как 1/0 для краткости
        freqs[1] = len(results) - freqs[0] # Оптимизация расчета для 2 интервалов
    elif k > 2:
        for res in results:
            if res < boundaries[0]: freqs[0] += 1
            elif res >= boundaries[-1]: freqs[-1] += 1
            else:
                for i in range(k - 2):
                    if boundaries[i] <= res < boundaries[i+1]:
                        freqs[i+1] += 1; break
    else: raise ValueError("k должно быть > 1") # Для ясности
    return freqs

def calculate_chi_squared_statistic(observed_frequencies, expected_frequencies):
    """Рассчитывает статистику хи-квадрат."""
    chi2_stat = 0
    for obs, exp in zip(observed_frequencies, expected_frequencies):
        if exp > 0: # Важная проверка, но теперь более краткая
            chi2_stat += (obs - exp) ** 2 / exp
    return chi2_stat

def perform_chi_squared_test(chi_squared_statistic, degrees_of_freedom, alpha):
    """Проводит тест хи-квадрат."""
    critical_value = stats.chi2.ppf(1 - alpha, degrees_of_freedom)
    p_value = 1 - stats.chi2.cdf(chi_squared_statistic, degrees_of_freedom)
    decision = "H0 принимается" if chi_squared_statistic < critical_value else "H0 отвергается" # Условное выражение для краткости
    return critical_value, p_value, decision

def print_hypothesis_test_results(chi_squared_statistic, p_value, critical_value, alpha, decision):
    """Выводит результаты теста хи-квадрат."""
    print("\nРезультаты теста Хи-квадрат:")
    print(f"Статистика χ² = {chi_squared_statistic:.4f}")
    print(f"Критическое значение = {critical_value:.4f}")
    print(f"P-значение = {p_value:.4f}")
    print(f"Уровень значимости α = {alpha:.4f}")
    print(f"Решение: {decision}")

def main():
    """Главная функция."""
    print("Моделирование потока вызовов Пуассона.") # Краткое приветствие

    while True:
        try: lambd = float(input("Введите интенсивность λ (> 0): ")); if lambd <= 0: raise ValueError
        except ValueError: print("Некорректный ввод. Введите положительное число.")
        else: break # Упрощенная структура цикла ввода

    while True:
        try: t = float(input("Введите время наблюдения t (> 0): ")); if t <= 0: raise ValueError
        except ValueError: print("Некорректный ввод. Введите положительное число.")
        else: break

    while True:
        try: num_simulations = int(input("Введите кол-во розыгрышей (> 0): ")); if num_simulations <= 0: raise ValueError
        except ValueError: print("Некорректный ввод. Введите целое число > 0.")
        else: break

    k_intervals, interval_boundaries = get_interval_boundaries()
    theoretical_interval_probs = calculate_theoretical_interval_probabilities(lambd, t, k_intervals, interval_boundaries)

    print("\nТеоретические вероятности интервалов:")
    for i, prob in enumerate(theoretical_interval_probs): # enumerate для краткости и ясности
        print(f"Интервал {i+1}: P(Δ{i+1}) = {prob:.4f}")

    while True:
        try: alpha = float(input("Введите уровень значимости α (0 < α < 1): ")); if not 0 < alpha < 1: raise ValueError
        except ValueError: print("Некорректный ввод. Введите число от 0 до 1.")
        else: break

    results = poisson_random_variable(lambd, t, num_simulations)
    frequency_table = create_frequency_table(np.array(results))
    characteristics = calculate_characteristics(results, lambd, t)
    goodness_of_fit_data = goodness_of_fit_table(frequency_table, lambd, t)
    kolmogorov_distance = kolmogorov_smirnov_test(results, lambd, t)

    print("\nРезультаты розыгрыша:")
    print_frequency_table(frequency_table)

    print("\nЧисловые характеристики:")
    print("Eη\tx\t|Eη - x|\tDη\tS^2\t|Dη - S^2|\tMe\tRb")
    char = characteristics # Сокращение имени для удобства
    print(f"{char['Eη']:.4f}\t{char['x']:.4f}\t{char['abs_diff_mean']:.4f}\t{char['Dη']:.4f}\t{char['S^2']:.4f}\t{char['abs_diff_variance']:.4f}\t{char['Me']:.4f}\t{char['Rb']:.4f}")

    print("\nТаблица согласия (макс. отклонение частот):") # Уточнение заголовка
    print_goodness_of_fit_table(goodness_of_fit_data)
    print(f"\nРасстояние Колмогорова-Смирнова: {kolmogorov_distance:.4f}")

    observed_interval_freqs = calculate_observed_interval_frequencies(results, k_intervals, interval_boundaries)
    expected_interval_freqs = [prob * num_simulations for prob in theoretical_interval_probs]
    chi_sq_stat = calculate_chi_squared_statistic(observed_interval_freqs, expected_interval_freqs)
    degrees_freedom = k_intervals - 1
    critical_val, p_val, hypothesis_decision = perform_chi_squared_test(chi_sq_stat, degrees_freedom, alpha)
    print_hypothesis_test_results(chi_sq_stat, p_val, critical_val, alpha, hypothesis_decision)

    plot_cdf(results, lambd, t)
    plt.hist(results, bins=range(min(results), max(results) + 2), align='left', rwidth=0.8)
    plt.xlabel("Число вызовов (η)")
    plt.ylabel("Частота")
    plt.title(f"Гистограмма Пуассона (λ={lambd}, t={t}, {num_simulations} симуляций)") # Сокращенный заголовок
    plt.xticks(range(min(results), max(results) + 1))
    plt.show()

if __name__ == "__main__":
    main()