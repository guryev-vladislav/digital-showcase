import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.path import Path

# Константа для площади, вычисленной интегрированием
EXACT_AREA = 6.28319

def calculate_area(num_samples):
    """
    Вычисляет площадь области, ограниченной кривой r = 2*cos(t), phi = sin(t),
    с использованием интегрирования и метода Монте-Карло, а также оценивает погрешность.
    """

    # Создаем массив значений t
    t = np.linspace(-np.pi, np.pi, 500)

    # Вычисляем координаты в декартовой системе
    x = 2 * np.cos(t) * np.cos(np.sin(t))
    y = 2 * np.cos(t) * np.sin(np.sin(t))

    # Ограничивающий прямоугольник
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    S_rect = (x_max - x_min) * (y_max - y_min)

    # Монте-Карло
    x_rand = np.random.uniform(x_min, x_max, num_samples)
    y_rand = np.random.uniform(y_min, y_max, num_samples)
    points = np.column_stack([x_rand, y_rand])

    # Создаем объект Path из точек кривой
    path = Path(np.column_stack([x, y]))

    # Проверяем, какие точки находятся внутри Path
    inside = path.contains_points(points)
    N_in = np.sum(inside)

    # Вычисление площади Монте-Карло
    area_monte_carlo = S_rect * (N_in / num_samples)

    # Оценка точности
    error = abs(area_monte_carlo - EXACT_AREA)
    relative_error = (error / EXACT_AREA) * 100

    return EXACT_AREA, area_monte_carlo, error, relative_error, x_rand[inside], y_rand[inside], x_rand[~inside], y_rand[~inside], x, y

def main():
    """Главная функция программы."""

    # Запрашиваем данные у пользователя
    num_samples = int(input("Введите количество точек Монте-Карло: "))
    num_iterations = int(input("Введите количество итераций: "))

    absolute_errors = []
    relative_errors = []

    for i in range(num_iterations):
        area, area_monte_carlo, error, relative_error, x_in, y_in, x_out, y_out, x, y = calculate_area(num_samples)
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
    plt.figure(figsize=(10, 10))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))

    plt.scatter(x_in, y_in, color='green', s=1, label='Внутри')
    plt.scatter(x_out, y_out, color='red', s=1, label='Снаружи')
    plt.plot(x, y, color='blue', linewidth=2, label='Кривая')

    plt.title('Метод Монте-Карло для вычисления площади')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()