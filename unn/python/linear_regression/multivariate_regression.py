import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS
import statsmodels.formula.api as smf

def multivariate_linear_regression_analysis(a1_true, a2_true, b_true, sigma_sq, n, m, t1, t2, s1, s2):
    """
    Выполняет анализ многомерной линейной регрессии.

    Args:
        a1_true (float): Истинный коэффициент при x1.
        a2_true (float): Истинный коэффициент при x2.
        b_true (float): Истинный коэффициент сдвига.
        sigma_sq (float): Дисперсия случайных ошибок.
        n (int): Размер первой выборки.
        m (int): Размер дополнительной выборки.
        t1 (float): Левая граница равномерного распределения для x1.
        t2 (float): Правая граница равномерного распределения для x1.
        s1 (float): Левая граница равномерного распределения для x2.
        s2 (float): Правая граница равномерного распределения для x2.

    Returns:
        tuple: Кортеж, содержащий:
            - results: Объект результатов регрессии от statsmodels.
            - y_predicted_additional (np.ndarray): Предсказанные значения для дополнительной выборки.
            - y_additional (np.ndarray): Истинные значения для дополнительной выборки.
    """
    # Шаг 1: Ввести коэффициенты и получить первую выборку
    x1 = np.random.uniform(t1, t2, n)
    x2 = np.random.uniform(s1, s2, n)
    epsilon = np.random.normal(0, np.sqrt(sigma_sq), n)
    y = a1_true * x1 + a2_true * x2 + b_true + epsilon

    # Создание DataFrame для statsmodels
    data = {'y': y, 'x1': x1, 'x2': x2}
    import pandas as pd
    df = pd.DataFrame(data)

    # Шаг 2: Оценить коэффициенты линейной регрессии
    formula = 'y ~ x1 + x2'
    model = smf.ols(formula, df).fit()
    results = model.summary()
    print("\nРезультаты регрессионной модели:\n", results)

    a1_estimated = model.params['x1']
    a2_estimated = model.params['x2']
    b_estimated = model.params['Intercept']
    r_squared = model.rsquared

    print(f"\nОцененный коэффициент при x1 (a1*): {a1_estimated:.4f}")
    print(f"Оцененный коэффициент при x2 (a2*): {a2_estimated:.4f}")
    print(f"Оцененный коэффициент сдвига (b*): {b_estimated:.4f}")
    print(f"Коэффициент детерминации (R^2): {r_squared:.4f}")

    # Шаг 4: Получить дополнительную выборку и сравнить предсказанные значения
    x1_additional = np.random.uniform(t1, t2, m)
    x2_additional = np.random.uniform(s1, s2, m)
    epsilon_additional = np.random.normal(0, np.sqrt(sigma_sq), m)
    y_additional = a1_true * x1_additional + a2_true * x2_additional + b_true + epsilon_additional
    y_predicted_additional = a1_estimated * x1_additional + a2_estimated * x2_additional + b_estimated

    print("\nСравнение предсказанных и истинных значений для дополнительной выборки:")
    for i in range(m):
        print(f"x1 = {x1_additional[i]:.4f}, x2 = {x2_additional[i]:.4f}, y_истинное = {y_additional[i]:.4f}, y_предсказанное = {y_predicted_additional[i]:.4f}")

    # Визуализация результатов (частично, так как 3D график)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, c='blue', marker='o', label='Первая выборка (обучающая)')

    # Создание плоскости оцененной регрессии
    x1_surf = np.linspace(min(x1), max(x1), 100)
    x2_surf = np.linspace(min(x2), max(x2), 100)
    X1, X2 = np.meshgrid(x1_surf, x2_surf)
    Y_predicted_surf = a1_estimated * X1 + a2_estimated * X2 + b_estimated
    ax.plot_surface(X1, X2, Y_predicted_surf, color='red', alpha=0.5, label='Оцененная плоскость регрессии')

    # Дополнительная выборка
    ax.scatter(x1_additional, x2_additional, y_additional, c='green', marker='^', label='Дополнительная выборка (тестовая)')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title('Многомерная линейная регрессия')
    ax.legend()
    plt.show()

    return model, y_predicted_additional, y_additional

if __name__ == "__main__":
    # Запрашиваем параметры у пользователя
    a1_true = float(input("Введите истинный коэффициент a1: "))
    a2_true = float(input("Введите истинный коэффициент a2: "))
    b_true = float(input("Введите истинный коэффициент b: "))
    sigma_sq = float(input("Введите дисперсию случайных ошибок (sigma^2): "))
    n = int(input("Введите размер первой выборки (n): "))
    m = int(input("Введите размер дополнительной выборки (m): "))
    t1 = float(input("Введите левую границу для x1 (t1): "))
    t2 = float(input("Введите правую границу для x1 (t2): "))
    s1 = float(input("Введите левую границу для x2 (s1): "))
    s2 = float(input("Введите правую границу для x2 (s2): "))

    # Запускаем анализ многомерной линейной регрессии
    regression_model, y_pred_add, y_add = multivariate_linear_regression_analysis(
        a1_true, a2_true, b_true, sigma_sq, n, m, t1, t2, s1, s2
    )