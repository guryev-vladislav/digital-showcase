import matplotlib.pyplot as plt
import random
import math
import numpy as np

def manual_det_3x3(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
            M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
            M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))

def solve_linear_system_cramer_manual_no_numpy(A, B):
    det_A = manual_det_3x3(A)
    if abs(det_A) < 1e-9:
        return None  # Матрица вырожденная

    n = len(A)
    solutions = [0.0] * n

    for i in range(n):
        Ai = [row[:] for row in A]
        for j in range(n):
            Ai[j][i] = B[j]
        solutions[i] = manual_det_3x3(Ai) / det_A

    return solutions

def multivariate_linear_regression_analysis_manual_no_numpy(a1_true, a2_true, b_true, sigma_sq, n, m, t1, t2, s1, s2):
    # Шаг 1: Ввести коэффициенты и получить первую выборку
    x1 = [random.uniform(t1, t2) for _ in range(n)]
    x2 = [random.uniform(s1, s2) for _ in range(n)]
    epsilon = [random.gauss(0, math.sqrt(sigma_sq)) for _ in range(n)]
    y = [a1_true * x1[i] + a2_true * x2[i] + b_true + epsilon[i] for i in range(n)]

    # Шаг 2: Оценить коэффициенты линейной регрессии вручную через систему уравнений
    sum_x1 = sum(x1)
    sum_x2 = sum(x2)
    sum_y = sum(y)
    sum_x1_sq = sum(xi**2 for xi in x1)
    sum_x2_sq = sum(xi**2 for xi in x2)
    sum_x1_x2 = sum(x1[i] * x2[i] for i in range(n))
    sum_x1_y = sum(x1[i] * y[i] for i in range(n))
    sum_x2_y = sum(x2[i] * y[i] for i in range(n))

    # Матрица коэффициентов системы уравнений
    A = [[n, sum_x1, sum_x2],
         [sum_x1, sum_x1_sq, sum_x1_x2],
         [sum_x2, sum_x1_x2, sum_x2_sq]]

    # Вектор правых частей системы уравнений
    B = [sum_y, sum_x1_y, sum_x2_y]

    coefficients = solve_linear_system_cramer_manual_no_numpy(A, B)

    if coefficients is None:
        print("Система линейных уравнений не имеет единственного решения (матрица вырожденная).")
        return None, None, None, None
    else:
        b_estimated = coefficients[0]
        a1_estimated = coefficients[1]
        a2_estimated = coefficients[2]

    # Шаг 3: Вычислить коэффициент детерминации R^2 вручную
    y_predicted = [a1_estimated * x1[i] + a2_estimated * x2[i] + b_estimated for i in range(n)]
    y_mean = sum(y) / n
    SSE = sum((y[i] - y_predicted[i])**2 for i in range(n))
    SST = sum((y[i] - y_mean)**2 for i in range(n))

    if SST == 0:
        r_squared = 0
    else:
        r_squared = 1 - (SSE / SST)

    print(f"\nОцененный коэффициент при x1 (a1*): {a1_estimated:.4f}")
    print(f"\nОцененный коэффициент при x2 (a2*): {a2_estimated:.4f}")
    print(f"\nОцененный коэффициент сдвига (b*): {b_estimated:.4f}")
    print(f"\nКоэффициент детерминации (R^2): {r_squared:.4f}")

    # Шаг 4: Получить дополнительную выборку и сравнить предсказанные значения
    x1_additional = [random.uniform(t1, t2) for _ in range(m)]
    x2_additional = [random.uniform(s1, s2) for _ in range(m)]
    epsilon_additional = [random.gauss(0, math.sqrt(sigma_sq)) for _ in range(m)]
    y_additional = [a1_true * x1_additional[i] + a2_true * x2_additional[i] + b_true + epsilon_additional[i] for i in range(m)]
    y_predicted_additional = [a1_estimated * x1_additional[i] + a2_estimated * x2_additional[i] + b_estimated for i in range(m)]

    # Визуализация результатов (частично, так как 3D график)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, c='blue', marker='o', label='Первая выборка (обучающая)')

    # Создание плоскости оцененной регрессии
    from mpl_toolkits.mplot3d import Axes3D
    x1_surf = np.linspace(min(x1), max(x1), 100)
    x2_surf = np.linspace(min(x2), max(x2), 100)
    X1_surf, X2_surf = np.meshgrid(x1_surf, x2_surf)
    Y_predicted_surf = a1_estimated * X1_surf + a2_estimated * X2_surf + b_estimated
    ax.plot_surface(X1_surf, X2_surf, Y_predicted_surf, color='red', alpha=0.5, label='Оцененная плоскость регрессии')

    # Дополнительная выборка
    ax.scatter(x1_additional, x2_additional, y_additional, c='green', marker='^', label='Дополнительная выборка (тестовая)')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title('Многомерная линейная регрессия (ручной расчет - без numpy)')
    ax.legend()

    # Добавление текстовой аннотации с начальными параметрами и R^2
    textstr = '\n'.join((
        f'a1_true = {a1_true:.2f}',
        f'a2_true = {a2_true:.2f}',
        f'b_true = {b_true:.2f}',
        f'sigma_sq = {sigma_sq:.2f}',
        f'n = {n}',
        f'R^2 = {r_squared:.2f}'
    ))

    # Эти координаты определяют положение текста
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)

    plt.show()

    return (a1_estimated, a2_estimated, b_estimated), y_predicted_additional, y_additional

if __name__ == "__main__":
    # Запрашиваем параметры у пользователя
    a1_true = 2.0
    a2_true = -1.0
    b_true = 5.0
    sigma_sq = 5
    n = 100
    m = 50
    t1 = 0.0
    t2 = 10.0
    s1 = -5.0
    s2 = 5.0

    # Запускаем анализ многомерной линейной регрессии (ручной расчет - без numpy)
    results, y_pred_add, y_add = multivariate_linear_regression_analysis_manual_no_numpy(
        a1_true, a2_true, b_true, sigma_sq, n, m, t1, t2, s1, s2
    )