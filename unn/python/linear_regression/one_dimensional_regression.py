import numpy as np
import matplotlib.pyplot as plt

def calculate_linear_regression(x, y):
    """Вычисляет параметры линейной регрессии."""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)

    if denominator == 0:
        a_estimated = 0
    else:
        a_estimated = numerator / denominator

    b_estimated = y_mean - a_estimated * x_mean

    y_predicted = a_estimated * x + b_estimated
    SSE = np.sum((y - y_predicted)**2)
    SST = np.sum((y - y_mean)**2)

    if SST == 0:
        r_squared = 0
    else:
        r_squared = 1 - (SSE / SST)

    return a_estimated, b_estimated, r_squared, y_predicted

def visualize_linear_regression(x_train, y_train, y_predicted_train, x_test=None, y_test=None, y_predicted_test=None, a_estimated=None, b_estimated=None, r_squared=None, t1=None, t2=None):
    """Визуализирует результаты линейной регрессии."""
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, label='Обучающие данные')
    if a_estimated is not None and b_estimated is not None:
        plt.plot(x_train, y_predicted_train, 'r--', label=f'Оцененная линия: ŷ = {a_estimated:.2f}x + {b_estimated:.2f}')
    if x_test is not None and y_test is not None:
        plt.scatter(x_test, y_test, color='orange', marker='o', label='Тестовые данные')
        if y_predicted_test is not None:
            plt.scatter(x_test, y_predicted_test, color='black', marker='x', label='Предсказанные значения на тесте') # Отображаем точки 'x'
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Линейная регрессия')
    plt.legend()
    plt.grid(True)

    text_parts = [
        f'n = {len(x_train)}'
    ]
    if x_test is not None:
        text_parts.append(f'm = {len(x_test)}')
    if r_squared is not None:
        text_parts.append(f'R^2 = {r_squared:.2f}')
    if t1 is not None and t2 is not None:
        text_parts.append(f'x_min = {t1:.2f}')
        text_parts.append(f'x_max = {t2:.2f}')

    text = '\n'.join(text_parts)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.show()

def linear_regression_analysis(a_true, b_true, sigma_sq, n, m, t1=0, t2=0):
    """
    Выполняет анализ линейной регрессии, генерируя синтетические данные и визуализируя результат.
    Эта функция теперь использует calculate_linear_regression и visualize_linear_regression.
    """
    if t1 == 0 and t2 == 0:
        x_train = np.arange(1, n + 1)
        x_test = np.arange(n + 1, n + m + 1)
    else:
        x_train = np.random.uniform(t1, t2, n)
        x_train.sort()
        x_test = np.random.uniform(t1, t2, m)
        x_test.sort()

    epsilon_train = np.random.normal(0, np.sqrt(sigma_sq), n)
    y_train = a_true * x_train + b_true + epsilon_train
    y_test_true = a_true * x_test + b_true

    a_estimated, b_estimated, r_squared, y_predicted_train = calculate_linear_regression(x_train, y_train)
    y_predicted_test = a_estimated * x_test + b_estimated

    print(f"Оцененный коэффициент наклона (a*): {a_estimated:.4f}")
    print(f"Оцененный коэффициент сдвига (b*): {b_estimated:.4f}")
    print(f"Коэффициент детерминации (R^2): {r_squared:.4f}")

    print("\nСравнение предсказанных и истинных значений для дополнительной выборки:")
    for i in range(m):
        print(f"x = {x_test[i]:.4f}, y_истинное = {y_test_true[i]:.4f}, y_предсказанное = {y_predicted_test[i]:.4f}")

    visualize_linear_regression(x_train, y_train, y_predicted_train, x_test, y_test_true, y_predicted_test, a_estimated, b_estimated, r_squared, t1, t2)

    return a_estimated, b_estimated, r_squared, y_predicted_test, y_test_true

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