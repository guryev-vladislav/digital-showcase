import numpy as np
import matplotlib.pyplot as plt
import collections
import math
import random
from scipy import stats

def poisson_random_variable(lambd, t, num_simulations):
    """Generates random numbers from a Poisson distribution using inverse transform sampling."""
    results = []
    for _ in range(num_simulations):
        u = random.random()  # Generate a uniform random number between 0 and 1
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
    """Creates a frequency table for a discrete random variable."""
    if len(data) == 0:  # Check for empty list - more pythonic than checking for numpy array size
        return {"yi": [], "ni": [], "ni/n": []}

    counter = collections.Counter(data)
    n = len(data)
    sorted_items = sorted(counter.items())

    yi = [item[0] for item in sorted_items]
    ni = [item[1] for item in sorted_items]
    ni_n = [item[1] / n for item in sorted_items]

    return {"yi": yi, "ni": ni, "ni/n": ni_n}

def print_frequency_table(frequency_table):
    """Prints the frequency table in the specified format."""
    if not frequency_table["yi"]:
        print("No data to display. Try increasing the number of simulations.")
        return

    num_rows = len(frequency_table["yi"])
    print("yi\tni\tni/n")  # Header
    for i in range(num_rows):
        print(f"{frequency_table['yi'][i]}\t{frequency_table['ni'][i]}\t{frequency_table['ni/n'][i]:.4f}")

def calculate_sample_variance(data):
    """Calculates the sample variance manually."""
    n = len(data)
    mean = sum(data) / n
    sum_sq_diff = sum((x - mean) ** 2 for x in data)
    return sum_sq_diff /n


def calculate_sample_median(data):
    """Calculates the sample median manually."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        mid1 = sorted_data[n // 2 - 1]
        mid2 = sorted_data[n // 2]
        median = (mid1 + mid2) / 2
    else:
        median = sorted_data[n // 2]
    return median

def calculate_characteristics(results, lambd, t):
    """Calculates numerical characteristics of the sample."""
    E_eta = lambd * t
    x_bar = np.mean(results)
    S_squared = calculate_sample_variance(results)  # Manual sample variance
    D_eta = E_eta
    median = calculate_sample_median(results)  # Manual sample median
    range_val = np.max(results) - np.min(results)

    return {
        "Eη": E_eta,
        "x": x_bar,
        "abs_diff_mean": abs(E_eta - x_bar),
        "Dη": D_eta,
        "S^2": S_squared,
        "abs_diff_variance": abs(D_eta - S_squared),
        "Me": median,
        "Rb": range_val
    }


def plot_cdf(results, lambd, t):
    """Plots the theoretical and empirical CDFs."""
    n = len(results)
    sorted_data = np.sort(results)
    empirical_cdf = np.arange(1, n + 1) / n
    empirical_cdf_shifted = empirical_cdf - 0.2

    # Theoretical CDF for Poisson distribution
    x_values = np.arange(max(results) + 1)
    theoretical_cdf = np.cumsum(stats.poisson.pmf(x_values, lambd * t))

    plt.step(sorted_data, empirical_cdf_shifted, label="Выборочная функция распределения (смещенная)")
    plt.step(x_values, theoretical_cdf, label="Теоретическая функция распределения")
    plt.xlabel("Число вызовов (η)")
    plt.ylabel("F(x)")
    plt.title("Функции распределения (выборочная и теоретическая)")
    plt.legend()
    plt.show()


def kolmogorov_smirnov_test(data, lambd, t):
    """Performs Kolmogorov-Smirnov test."""
    from scipy import stats
    n = len(data)
    x_values = np.arange(max(data) + 1)
    theoretical_cdf = np.cumsum(stats.poisson.pmf(x_values, lambd * t))

    # Find the maximum absolute difference between empirical and theoretical CDFs
    D = max(abs(np.interp(x_values, np.sort(data), np.arange(1, n + 1) / n) - theoretical_cdf))
    return D


def goodness_of_fit_table(frequency_table, lambd, t):
    """Generates the goodness of fit table."""
    yi = frequency_table['yi']
    ni = frequency_table['ni']
    n = sum(ni)
    theoretical_probs = [stats.poisson.pmf(y, lambd * t) for y in yi]
    ni_n = [i / n for i in ni]
    max_deviation = max([abs(a - b) for a, b in zip(ni_n, theoretical_probs)])

    table_data = {
        "yj": yi,
        "P({η = yj})": theoretical_probs,
        "nj/n": ni_n,
        "Max Deviation": max_deviation
    }

    return table_data


def print_goodness_of_fit_table(table_data):
    print("yj\tP({η=yj})\tnj/n")
    for i in range(len(table_data['yj'])):
        print(f"{table_data['yj'][i]}\t{table_data['P({η = yj})'][i]:.4f}\t{table_data['nj/n'][i]:.4f}")
    print(f"Максимальное отклонение: {table_data['Max Deviation']:.4f}")

def main():
    """Main function to handle user interaction and simulation."""

    print("Программа моделирует поток вызовов на АТС с распределением Пуассона.")
    print("Случайная величина η — число вызовов за t минут, имеет распределение Пуассона со средним λt.")

    while True:
        try:
            lambd = float(input("Введите интенсивность потока вызовов (λ): "))
            if lambd <= 0:
                raise ValueError("Интенсивность должна быть положительным числом.")
            break
        except ValueError:
            print("Некорректный ввод. Пожалуйста, введите число.")

    while True:
        try:
            t = float(input("Введите время наблюдения (t минут): "))
            if t <= 0:
                raise ValueError("Время наблюдения должно быть положительным числом.")
            break
        except ValueError:
            print("Некорректный ввод. Пожалуйста, введите число.")

    while True:
        try:
            num_simulations = int(input("Введите количество розыгрышей: "))
            if num_simulations <= 0:
                raise ValueError("Количество розыгрышей должно быть положительным целым числом.")
            break
        except ValueError:
            print("Некорректный ввод. Пожалуйста, введите целое число.")

    from scipy import stats
    results = poisson_random_variable(lambd, t, num_simulations)
    frequency_table = create_frequency_table(np.array(results))
    characteristics = calculate_characteristics(results, lambd, t)
    goodness_of_fit_data = goodness_of_fit_table(frequency_table, lambd, t)
    kolmogorov_distance = kolmogorov_smirnov_test(results, lambd, t)

    print("\nРезультаты розыгрыша:")
    print(results)
    print_frequency_table(frequency_table)

    print("\nЧисловые характеристики:")
    print("Eη\tx\t|Eη - x|\tDη\tS^2\t|Dη - S^2|\tMe\tRb")
    print(
        f"{characteristics['Eη']:.4f}\t{characteristics['x']:.4f}\t{characteristics['abs_diff_mean']:.4f}\t{characteristics['Dη']:.4f}\t{characteristics['S^2']:.4f}\t{characteristics['abs_diff_variance']:.4f}\t{characteristics['Me']:.4f}\t{characteristics['Rb']:.4f}")

    print("\nТаблица согласия:")
    print_goodness_of_fit_table(goodness_of_fit_data)

    print(f"\nРасстояние Колмогорова-Смирнова: {kolmogorov_distance:.4f}")

    plot_cdf(results, lambd, t)
    plt.hist(results, bins=range(min(results), max(results) + 2), align='left', rwidth=0.8)
    plt.xlabel("Число вызовов (η)")
    plt.ylabel("Частота")
    plt.title(f"Гистограмма распределения Пуассона (λ={lambd}, t={t}, {num_simulations} simulations)")
    plt.xticks(range(min(results), max(results) + 1))
    plt.show()


if __name__ == "__main__":
    main()

