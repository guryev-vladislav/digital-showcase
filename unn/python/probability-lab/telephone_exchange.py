import numpy as np
import matplotlib.pyplot as plt
import collections
import math
import random

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

    results = poisson_random_variable(lambd, t, num_simulations)
    frequency_table = create_frequency_table(np.array(results))  # Convert list to numpy array here

    print("\nРезультаты розыгрыша:")
    print(results)  # Вывод сырых данных

    print_frequency_table(frequency_table)

    # Гистограмма (опционально)
    plt.hist(results, bins=range(min(results), max(results) + 2), align='left', rwidth=0.8)
    plt.xlabel("Число вызовов (η)")
    plt.ylabel("Частота")
    plt.title(f"Гистограмма распределения Пуассона (λ={lambd}, t={t}, {num_simulations} simulations)")
    plt.xticks(range(min(results), max(results) + 1))
    plt.show()


if __name__ == "__main__":
    main()

