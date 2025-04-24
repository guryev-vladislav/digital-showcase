import numpy as np
import matplotlib.pyplot as plt

def linear_regression_analysis(a_true, b_true, sigma_sq, n, m, t1=0, t2=0):
    """
    Выполняет анализ одномерной линейной регрессии.

    Args:
        a_true (float): Истинный коэффициент наклона.
        b_true (float): Истинный коэффициент сдвига.
        sigma_sq (float): Дисперсия случайных ошибок.
        n (int): Размер первой выборки.
        m (int): Размер дополнительной выборки.
        t1 (float): Левая граница отрезка для случайного выбора x (по умолчанию 0).
        t2 (float): Правая граница отрезка для случайного выбора x (по умолчанию 0).

    Returns:
        tuple: Кортеж, содержащий:
            - a_estimated (float): Оцененный коэффициент наклона.
            - b_estimated (float): Оцененный коэффициент сдвига.
            - r_squared (float): Коэффициент детерминации R^2.
            - y_predicted_additional (np.ndarray): Предсказанные значения для дополнительной выборки (по оцененной линии).
            - y_additional (np.ndarray): Истинные значения для дополнительной выборки (на истинной линии).
    """
    # Ввести коэффициенты и получить первую выборку
    if t1 == 0 and t2 == 0:
        x = np.arange(1, n + 1)
    else:
        x = np.random.uniform(t1, t2, n)
        x.sort() # Для более наглядного графика
    epsilon = np.random.normal(0, np.sqrt(sigma_sq), n)
    y = a_true * x + b_true + epsilon

    # Оценить коэффициенты линейной регрессии вручную
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)

    if denominator == 0:
        a_estimated = 0
    else:
        a_estimated = numerator / denominator

    b_estimated = y_mean - a_estimated * x_mean

    # Вычислить коэффициент детерминации R^2 вручную
    y_predicted = a_estimated * x + b_estimated
    SSE = np.sum((y - y_predicted)**2)
    SST = np.sum((y - y_mean)**2)

    if SST == 0:
        r_squared = 0
    else:
        r_squared = 1 - (SSE / SST)

    print(f"Оцененный коэффициент наклона (a*): {a_estimated:.4f}")
    print(f"Оцененный коэффициент сдвига (b*): {b_estimated:.4f}")
    print(f"Коэффициент детерминации (R^2): {r_squared:.4f}")

    # Получить дополнительную выборку и сравнить предсказанные значения
    if t1 == 0 and t2 == 0:
        x_additional = np.arange(n + 1, n + m + 1)
    else:
        x_additional = np.random.uniform(t1, t2, m)
        x_additional.sort() # Для более наглядного графика
    # Генерация дополнительной выборки ТОЧНО на истинной линии
    y_additional = a_true * x_additional + b_true
    y_predicted_additional = a_estimated * x_additional + b_estimated

    print("\nСравнение предсказанных и истинных значений для дополнительной выборки:")
    for i in range(m):
        print(f"x = {x_additional[i]:.4f}, y_истинное = {y_additional[i]:.4f}, y_предсказанное = {y_predicted_additional[i]:.4f}")

    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Первая выборка (обучающая)')
    plt.plot(x, a_true * x + b_true, 'g-', label=f'Истинная линия: y = {a_true}x + {b_true}')
    plt.plot(x, y_predicted, 'r--', label=f'Оцененная линия: ŷ = {a_estimated:.2f}x + {b_estimated:.2f}')
    plt.scatter(x_additional, y_additional, color='orange', marker='o', label='Дополнительная выборка (на истинной линии)')
    plt.plot(x_additional, y_predicted_additional, 'm--', label='Предсказанные значения для доп. выборки')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Одномерная линейная регрессия')
    plt.legend()
    plt.grid(True)

    # Добавление текстовой аннотации с истинными параметрами и R^2
    text = '\n'.join((
        f'true_a = {a_true:.2f}',
        f'true_b = {b_true:.2f}',
        f'sigma = {np.sqrt(sigma_sq):.2f}',
        f'n = {n}',
        f'm = {m}',
        f'R^2 = {r_squared:.2f}',
        f't1 = {t1:.2f}',
        f't2 = {t2:.2f}'
    ))

    # Эти координаты определяют положение текста
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.show()

    return a_estimated, b_estimated, r_squared, y_predicted_additional, y_additional

if __name__ == "__main__":
    true_a = float(input("Введите истинный коэффициент a: "))
    true_b = float(input("Введите истинный коэффициент b: "))
    error_sigma_sq = float(input("Введите дисперсию случайных ошибок (sigma^2): "))
    n_samples = int(input("Введите размер первой выборки (n): "))
    m_additional_samples = int(input("Введите размер дополнительной выборки (m): "))
    t1_input = float(input("Введите левую границу для x (t1, 0 - по умолчанию): "))
    t2_input = float(input("Введите правую границу для x (t2, 0 - по умолчанию): "))


    # Запускаем анализ линейной регрессии
    estimated_a, estimated_b, r2, y_pred_add, y_add = linear_regression_analysis(
        true_a, true_b, error_sigma_sq, n_samples, m_additional_samples, t1_input, t2_input
    )