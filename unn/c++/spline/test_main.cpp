#include "SplineCalculator.h"
#include <iostream>
#include <vector>
#include <iomanip> // Для форматированного вывода

// Пример функции, которую будем аппроксимировать
double originalFunction(double x) {
    return x * x * x; // Пример: f(x) = x^3
}

// Пример производной функции
double originalFunctionDerivative(double x) {
    return 3 * x * x; // Пример: f'(x) = 3x^2
}

// Пример второй производной функции
double originalFunctionSecondDerivative(double x) {
    return 6 * x;     // Пример: f''(x) = 6x
}

int main() {
    // 1. Задаем исходные данные
    std::vector<double> x_data = { -1.0, 0.0, 1.0, 2.0, 3.0 };
    std::vector<double> y_data;
    for (double x : x_data) {
        y_data.push_back(originalFunction(x)); // Вычисляем значения функции в точках x
    }

    // 2. Создаем объект SplineCalculator, передавая функции производных
    SplineCalculator calculator(x_data, y_data, originalFunctionDerivative, originalFunctionSecondDerivative);

    // 3. Вычисляем коэффициенты сплайна
    std::vector<double> spline_coefficients = calculator.solve();

    // Вывод коэффициентов (для проверки)
    std::cout << "Spline Coefficients (m):" << std::endl;
    for (double coeff : spline_coefficients) {
        std::cout << std::fixed << std::setprecision(4) << coeff << " ";
    }
    std::cout << std::endl;

    // 4. Задаем точки для оценки сплайна и погрешностей
    std::vector<double> x_evaluation_points;
    for (double x = x_data.front() - 0.5; x <= x_data.back() + 0.5; x += 0.25) {
        x_evaluation_points.push_back(x);
    }

    // 5. Вычисляем значения сплайна, погрешности и их производные
    std::vector<double> spline_values = calculator.count_S(x_evaluation_points);
    std::vector<double> errors = calculator.count_pogr(x_evaluation_points, y_data);
    std::vector<double> derivative_errors = calculator.count_pogr_pr1(x_evaluation_points, y_data);
    std::vector<double> second_derivative_errors = calculator.count_pogr_pr2(x_evaluation_points, y_data);

    // 6. Выводим результаты
    std::cout << "\nEvaluation Results:" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "  x   |  f(x)  |  S(x)  | Error | Error' | Error''" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    for (size_t i = 0; i < x_evaluation_points.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4)
                  << x_evaluation_points[i] << " | "
                  << originalFunction(x_evaluation_points[i]) << " | "
                  << spline_values[i] << " | "
                  << errors[i] << " | "
                  << derivative_errors[i] << " | "
                  << second_derivative_errors[i] << std::endl;
    }
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    // 7. Выводим максимальные погрешности
    std::vector<std::pair<double, int>> max_error_positions = calculator.findMax(errors);
    std::vector<std::pair<double, int>> max_derivative_error_positions = calculator.findMax(derivative_errors);
    std::vector<std::pair<double, int>> max_second_derivative_error_positions = calculator.findMax(second_derivative_errors);

    std::cout << "\nMaximum Errors:" << std::endl;
    std::cout << "Max |f(x) - S(x)|: " << std::fixed << std::setprecision(4) << max_error_positions[0].first
              << " at x = " << x_evaluation_points[max_error_positions[0].second] << std::endl;
    std::cout << "Max |f'(x) - S'(x)|: " << std::fixed << std::setprecision(4) << max_derivative_error_positions[0].first
              << " at x = " << x_evaluation_points[max_derivative_error_positions[0].second] << std::endl;
    std::cout << "Max |f''(x) - S''(x)|: " << std::fixed << std::setprecision(4) << max_second_derivative_error_positions[0].first
              << " at x = " << x_evaluation_points[max_second_derivative_error_positions[0].second] << std::endl;

    return 0;
}