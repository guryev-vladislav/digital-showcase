import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from one_dimensional_regression import calculate_linear_regression, visualize_linear_regression

if __name__ == "__main__":
    try:
        df = pd.read_csv('Advertising.csv')
        # Замените 'TV' и 'Sales' на названия столбцов в вашем файле
        x_real = df['TV'].tolist()
        y_real = df['sales'].tolist()

        n_samples = int(input("Введите размер первой выборки (n): "))
        m_additional_samples = int(input("Введите размер дополнительной выборки (m): "))

        # Возьмем первые n_samples для обучения
        x_train = np.array(x_real[:n_samples])
        y_train = np.array(y_real[:n_samples])

        # Оценим диапазон x для передачи в визуализацию
        t1_real = np.min(x_train)
        t2_real = np.max(x_train)

        # Вычисляем параметры регрессии
        estimated_a, estimated_b, r2, y_predicted_train = calculate_linear_regression(x_train, y_train)

        x_test = None
        y_test = None
        y_predicted_test = None

        if m_additional_samples > 0 and len(x_real) > n_samples:
            x_test = np.array(x_real[n_samples:n_samples + m_additional_samples])
            y_test = np.array(y_real[n_samples:n_samples + m_additional_samples])
            y_predicted_test = estimated_a * x_test + estimated_b

            print("\nСравнение предсказанных и реальных значений для дополнительной выборки:")
            for i in range(len(x_test)):
                print(f"x = {x_test[i]:.4f}, y_реальное = {y_test[i]:.4f}, y_предсказанное = {y_predicted_test[i]:.4f}")

        # Визуализируем результаты
        visualize_linear_regression(x_train, y_train, y_predicted_train, x_test, y_test, y_predicted_test, estimated_a, estimated_b, r2, t1_real, t2_real)

    except FileNotFoundError:
        print("Ошибка: Файл 'advertising.csv' не найден. Пожалуйста, убедитесь, что файл находится в той же директории, что и скрипт, или укажите правильный путь к файлу.")
    except KeyError as e:
        print(f"Ошибка: Не найден столбец '{e.args[0]}' в CSV файле. Пожалуйста, убедитесь, что названия столбцов 'TV' и 'Sales' соответствуют вашему файлу.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")