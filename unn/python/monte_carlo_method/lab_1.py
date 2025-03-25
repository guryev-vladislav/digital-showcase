import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def calculate_area(num_samples):
    """
    Вычисляет площадь области, ограниченной кривой r = 2*cos(t), phi = sin(t),
    с использованием интегрирования и метода Монте-Карло, а также оценивает погрешность.

    Args:
        num_samples (int): Количество случайных точек для метода Монте-Карло.

    Returns:
        tuple: Кортеж, содержащий:
            - area (float): Площадь, вычисленная интегрированием.
            - area_monte_carlo (float): Площадь, вычисленная методом Монте-Карло.
            - error (float): Абсолютная погрешность между двумя методами.
            - relative_error (float): Относительная погрешность (в процентах).
        tuple:  `x_rand`, `y_rand`, `inside_flag_array`, `x`, `y`: случайные точки для отрисовки графика
    """

    # Создаем массив значений t
    t = np.linspace(-np.pi, np.pi, 400)

    # Вычисляем координаты в полярной системе
    r = 2 * np.cos(t)
    phi = np.sin(t)

    # Преобразуем в декартовы координаты
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # Вычисление площади области, ограниченной кривой (интегрированием)
    def integrand(t):
        return (2 * np.cos(t))**2 / 2

    area, _ = quad(integrand, -np.pi, np.pi)

    # Метод Монте-Карло для вычисления площади
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2

    x_rand = np.random.uniform(x_min, x_max, num_samples)
    y_rand = np.random.uniform(y_min, y_max, num_samples)

    inside_count = 0
    inside_flag_array = []
    for i in range(num_samples):
        x_point = x_rand[i]
        y_point = y_rand[i]
        r_point = np.sqrt(x_point**2 + y_point**2)
        t_point = np.arctan2(y_point, x_point)

        if r_point <= np.abs(2 * np.cos(t_point)):
            inside_count += 1
            inside_flag_array.append('green')
        else:
            inside_flag_array.append('red')




    # Вычисление площади через Монте-Карло
    area_monte_carlo = (inside_count / num_samples) * (x_max - x_min) * (y_max - y_min)

    # Оценка точности
    error = abs(area_monte_carlo - area)
    relative_error = (error / area) * 100

    return area, area_monte_carlo, error, relative_error, x_rand, y_rand, inside_flag_array, x, y


def main():
    """
    Главная функция программы. Инициализирует параметры и вызывает функцию для вычисления площади.
    """
    num_samples = 10000  # Количество точек для Монте-Карло
    num_iterations = 1 # Количество итераций Монте-Карло

    absolute_errors = []
    relative_errors = []
    area = 0 # Инициализация переменной area
    x_rand, y_rand, inside_flag_array, x, y = [], [], [], [], [] # Инициализация переменных для графика

    # Запускаем Монте-Карло несколько раз и выводим результаты
    for i in range(num_iterations):
        #print(f"\nИтерация Монте-Карло #{i+1}")
        area, area_monte_carlo, error, relative_error, x_rand, y_rand, inside_flag_array, x, y = calculate_area(num_samples)
        absolute_errors.append(error)
        relative_errors.append(relative_error)

        # Выводим результаты первой итерации
        if i == 0:
            print(f"Площадь области (интеграл): {area:.5f}")
            print(f"Площадь области (Монте-Карло): {area_monte_carlo:.5f}")
            print(f"Абсолютная погрешность: {error:.5f}")
            print(f"Относительная погрешность: {relative_error:.5f}%")


    # Вычисляем средние значения погрешностей
    mean_absolute_error = np.mean(absolute_errors)
    mean_relative_error = np.mean(relative_errors)

    print(f"\nСредняя абсолютная погрешность (по {num_iterations} итерациям): {mean_absolute_error:.5f}")
    print(f"Средняя относительная погрешность (по {num_iterations} итерациям): {mean_relative_error:.5f}%")

    # Строим график
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label=r'$r = 2\cos(t), \; phi = \sin(t)$')
    plt.scatter(x_rand,y_rand, color = inside_flag_array, s=1)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.title('Полярный график в декартовых координатах')
    plt.show()



if __name__ == "__main__":
    main()
