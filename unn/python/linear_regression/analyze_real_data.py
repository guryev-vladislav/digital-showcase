import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from one_dimensional_regression import calculate_linear_regression, visualize_linear_regression

def calculate_r_squared(y_true, y_predicted):
    """Вычисляет коэффициент детерминации R^2."""
    if len(y_true) != len(y_predicted):
        raise ValueError("Размеры y_true и y_predicted должны быть одинаковыми.")
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean)**2)
    ss_residual = np.sum((y_true - y_predicted)**2)
    if ss_total == 0:
        return 0
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

if __name__ == "__main__":
    try:
        df = pd.read_csv('Advertising.csv')
        x_real = df['TV'].tolist()
        y_real = df['sales'].tolist()

        n_samples = int(input(f"Введите размер первой выборки (n, макс. {len(x_real)}): "))
        m_additional_samples = int(input(f"Введите размер дополнительной выборки (m, макс. {len(x_real) - n_samples}): "))

        if n_samples > len(x_real):
            n_samples = len(x_real)
        if m_additional_samples > len(x_real) - n_samples:
            m_additional_samples = len(x_real) - n_samples

        x_train = np.array(x_real[:n_samples])
        y_train = np.array(y_real[:n_samples])

        x_test = None
        y_test = None
        if m_additional_samples > 0:
            x_test = np.array(x_real[n_samples:n_samples + m_additional_samples])
            y_test = np.array(y_real[n_samples:n_samples + m_additional_samples])

        t1_real = np.min(x_train)
        t2_real = np.max(x_train)

        # Линейная регрессия
        estimated_a_linear, estimated_b_linear, r2_linear_train, y_predicted_train_linear = calculate_linear_regression(x_train, y_train)
        y_predicted_test_linear = estimated_a_linear * x_test + estimated_b_linear if x_test is not None else None
        r2_linear_test = calculate_r_squared(y_test, y_predicted_test_linear) if x_test is not None else None

        # Полиномиальная регрессия 2-го порядка
        poly_coeffs_2 = np.polyfit(x_train, y_train, 2)
        p_2 = np.poly1d(poly_coeffs_2)
        y_predicted_test_poly_2 = p_2(x_test) if x_test is not None else None
        r2_poly_2 = calculate_r_squared(y_test, y_predicted_test_poly_2) if x_test is not None else None

        # Полиномиальная регрессия 3-го порядка
        poly_coeffs_3 = np.polyfit(x_train, y_train, 3)
        p_3 = np.poly1d(poly_coeffs_3)
        y_predicted_test_poly_3 = p_3(x_test) if x_test is not None else None
        r2_poly_3 = calculate_r_squared(y_test, y_predicted_test_poly_3) if x_test is not None else None

        print("\nРезультаты линейной регрессии (обучение):")
        print(f"Оцененный коэффициент наклона (a*): {estimated_a_linear:.4f}")
        print(f"Оцененный коэффициент сдвига (b*): {estimated_b_linear:.4f}")
        print(f"Коэффициент детерминации (R^2): {r2_linear_train:.4f}")

        if x_test is not None:
            print("\nРезультаты линейной регрессии (тест):")
            print(f"Коэффициент детерминации (R^2): {r2_linear_test:.4f}")
            print("\nРезультаты полиномиальной регрессии 2-го порядка (тест):")
            print(f"Коэффициент детерминации (R^2): {r2_poly_2:.4f}")
            print("\nРезультаты полиномиальной регрессии 3-го порядка (тест):")
            print(f"Коэффициент детерминации (R^2): {r2_poly_3:.4f}")

        # Визуализация
        plt.figure(figsize=(12, 8))
        plt.scatter(x_train, y_train, label='Обучающие данные')
        if x_test is not None:
            plt.scatter(x_test, y_test, color='orange', marker='o', label='Тестовые данные')
            sort_indices = np.argsort(x_test)
            plt.plot(x_test[sort_indices], y_predicted_test_linear[sort_indices], 'r--', label=f'Линейная регрессия (тест), R^2={r2_linear_test:.2f}')
            plt.plot(x_test[sort_indices], y_predicted_test_poly_2[sort_indices], 'g-', label=f'Полином 2 (тест), R^2={r2_poly_2:.2f}')
            plt.plot(x_test[sort_indices], y_predicted_test_poly_3[sort_indices], 'b-', label=f'Полином 3 (тест), R^2={r2_poly_3:.2f}')

        plt.xlabel('TV')
        plt.ylabel('Sales')
        plt.title('Сравнение линейной и полиномиальной регрессии на тестовых данных')
        plt.legend()
        plt.grid(True)

        text = '\n'.join((
            f'n = {len(x_train)}',
            f'm = {len(x_test) if x_test is not None else 0}',
            f'R^2 (линейная, обучение) = {r2_linear_train:.2f}',
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)

        plt.show()

    except FileNotFoundError:
        print("Ошибка: Файл 'Advertising.csv' не найден...")
    except KeyError as e:
        print(f"Ошибка: Не найден столбец '{e.args[0]}'...")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        print(f"Произошла ошибка: {e}")