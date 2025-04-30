import numpy as np
import matplotlib.pyplot as plt

def manual_det_3x3(M):
    """Вычисляет определитель матрицы 3x3 вручную."""
    return (M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) -
            M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0]) +
            M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]))

def solve_linear_system_cramer_manual(A, B):
    """Решает систему линейных уравнений Ax = B методом Крамера (вычисление определителя вручную)."""
    det_A = manual_det_3x3(A)
    if np.abs(det_A) < 1e-9:
        return None  # Матрица вырожденная

    n = A.shape[0]
    solutions = np.zeros(n)

    for i in range(n):
        Ai = np.copy(A)
        Ai[:, i] = B
        solutions[i] = manual_det_3x3(Ai) / det_A

    return solutions

def multivariate_linear_regression_analysis_manual_equations_cramer_manual_det(a1_true, a2_true, b_true, sigma_sq, n, m, t1, t2, s1, s2):
    """
    Выполняет анализ многомерной линейной регрессии (вычисление вручную через систему уравнений методом Крамера с ручным определителем).
    """
    # Шаг 1: Ввести коэффициенты и получить первую выборку
    x1 = np.random.uniform(t1, t2, n)
    x2 = np.random.uniform(s1, s2, n)
    epsilon = np.random.normal(0, np.sqrt(sigma_sq), n)
    y = a1_true * x1 + a2_true * x2 + b_true + epsilon

    # Шаг 2: Оценить коэффициенты линейной регрессии вручную через систему уравнений
    sum_x1 = np.sum(x1)
    sum_x2 = np.sum(x2)
    sum_y = np.sum(y)
    sum_x1_sq = np.sum(x1**2)
    sum_x2_sq = np.sum(x2**2)
    sum_x1_x2 = np.sum(x1 * x2)
    sum_x1_y = np.sum(x1 * y)
    sum_x2_y = np.sum(x2 * y)

    # Матрица коэффициентов системы уравнений
    A = np.array([[n, sum_x1, sum_x2],
                  [sum_x1, sum_x1_sq, sum_x1_x2],
                  [sum_x2, sum_x1_x2, sum_x2_sq]])

    # Вектор правых частей системы уравнений
    B = np.array([sum_y, sum_x1_y, sum_x2_y])

    coefficients = solve_linear_system_cramer_manual(A, B)

    if coefficients is None:
        print("Система линейных уравнений не имеет единственного решения (матрица вырожденная).")
        return None, None, None, None
    else:
        b_estimated = coefficients[0]
        a1_estimated = coefficients[1]
        a2_estimated = coefficients[2]

    # Шаг 3: Вычислить коэффициент детерминации R^2 вручную
    y_predicted = a1_estimated * x1 + a2_estimated * x2 + b_estimated
    y_mean = np.mean(y)
    SSE = np.sum((y - y_predicted)**2)
    SST = np.sum((y - y_mean)**2)

    if SST == 0:
        r_squared = 0
    else:
        r_squared = 1 - (SSE / SST)

    print(f"\nОцененный коэффициент при x1 (a1*): {a1_estimated:.4f}")
    print(f"\nОцененный коэффициент при x2 (a2*): {a2_estimated:.4f}")
    print(f"\nОцененный коэффициент сдвига (b*): {b_estimated:.4f}")
    print(f"\nКоэффициент детерминации (R^2): {r_squared:.4f}")

    # Шаг 4: Получить дополнительную выборку и сравнить предсказанные значения
    x1_additional = np.random.uniform(t1, t2, m)
    x2_additional = np.random.uniform(s1, s2, m)
    epsilon_additional = np.random.normal(0, np.sqrt(sigma_sq), m)
    y_additional = a1_true * x1_additional + a2_true * x2_additional + b_true + epsilon_additional
    y_predicted_additional = a1_estimated * x1_additional + a2_estimated * x2_additional + b_estimated

    # Визуализация результатов (частично, так как 3D график)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, c='blue', marker='o', label='Первая выборка (обучающая)')

    # Создание плоскости оцененной регрессии
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
    ax.set_title('Многомерная линейная регрессия (ручной расчет через уравнения - Крамер с ручным определителем)')
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

    return coefficients, y_predicted_additional, y_additional

if __name__ == "__main__":
    # Запрашиваем параметры у пользователя
    a1_true = 2.0
    a2_true = -1.0
    b_true = 5.0
    sigma_sq = 5
    n = 10000
    m = 50
    t1 = 0.0
    t2 = 10.0
    s1 = -5.0
    s2 = 5.0

    # Запускаем анализ многомерной линейной регрессии (ручной расчет через уравнения методом Крамера с ручным определителем)
    results, y_pred_add, y_add = multivariate_linear_regression_analysis_manual_equations_cramer_manual_det(
        a1_true, a2_true, b_true, sigma_sq, n, m, t1, t2, s1, s2
    )