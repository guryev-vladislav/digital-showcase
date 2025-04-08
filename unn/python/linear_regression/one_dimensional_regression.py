import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def linear_regression_analysis(a_true, b_true, sigma_sq, n, m):
    """
    Выполняет анализ одномерной линейной регрессии.

    Args:
        a_true (float): Истинный коэффициент наклона.
        b_true (float): Истинный коэффициент сдвига.
        sigma_sq (float): Дисперсия случайных ошибок.
        n (int): Размер первой выборки.
        m (int): Размер дополнительной выборки.

    Returns:
        tuple: Кортеж, содержащий:
            - a_estimated (float): Оцененный коэффициент наклона.
            - b_estimated (float): Оцененный коэффициент сдвига.
            - r_squared (float): Коэффициент детерминации R^2.
            - y_predicted_additional (np.ndarray): Предсказанные значения для дополнительной выборки.
            - y_additional (np.ndarray): Истинные значения для дополнительной выборки.
    """
    # Шаг 1: Ввести коэффициенты и получить первую выборку
    x = np.arange(1, n + 1)
    epsilon = np.random.normal(0, np.sqrt(sigma_sq), n)
    y = a_true * x + b_true + epsilon

    # Шаг 2: Оценить коэффициенты линейной регрессии
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    a_estimated = slope
    b_estimated = intercept
    r_squared = r_value**2

    # Шаг 3: Вычислить коэффициент детерминации R^2
    print(f"Оцененный коэффициент наклона (a*): {a_estimated:.4f}")
    print(f"Оцененный коэффициент сдвига (b*): {b_estimated:.4f}")
    print(f"Коэффициент детерминации (R^2): {r_squared:.4f}")

    # Шаг 4: Получить дополнительную выборку и сравнить предсказанные значения
    x_additional = np.arange(n + 1, n + m + 1)
    epsilon_additional = np.random.normal(0, np.sqrt(sigma_sq), m)
    y_additional = a_true * x_additional + b_true + epsilon_additional
    y_predicted_additional = a_estimated * x_additional + b_estimated

    print("\nСравнение предсказанных и истинных значений для дополнительной выборки:")
    for i in range(m):
        print(f"x = {x_additional[i]}, y_истинное = {y_additional[i]:.4f}, y_предсказанное = {y_predicted_additional[i]:.4f}")

    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Первая выборка (обучающая)')
    plt.plot(x, a_true * x + b_true, 'g-', label=f'Истинная линия: y = {a_true}x + {b_true}')
    plt.plot(x, a_estimated * x + b_estimated, 'r--', label=f'Оцененная линия: ŷ = {a_estimated:.2f}x + {b_estimated:.2f}')
    plt.scatter(x_additional, y_additional, label='Дополнительная выборка (тестовая)')
    plt.plot(x_additional, y_predicted_additional, 'm--', label='Предсказанные значения для доп. выборки')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Одномерная линейная регрессия')
    plt.legend()
    plt.grid(True)
    plt.show()

    return a_estimated, b_estimated, r_squared, y_predicted_additional, y_additional

if __name__ == "__main__":
    # Задаем истинные параметры и размеры выборок
    true_a = 2.5
    true_b = 1.0
    error_sigma_sq = 4.0
    n_samples = 50
    m_additional_samples = 20

    # Запускаем анализ линейной регрессии
    estimated_a, estimated_b, r2, y_pred_add, y_add = linear_regression_analysis(true_a, true_b, error_sigma_sq, n_samples, m_additional_samples)