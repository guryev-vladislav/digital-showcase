#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <clocale>

using namespace std;

// Функция для правой части уравнения
double f(double x, double y) {
    return -(4 - 4 * x * x - 4 * y * y) * exp(1 - x * x - y * y);
}

// Граничные условия
double mu1(double y) { return exp(-y * y); } // u(-1, y)
double mu2(double y) { return exp(-y * y); } // u(1, y)
double mu3(double x) { return exp(-x * x); } // u(x, -1)
double mu4(double x) { return exp(-x * x); } // u(x, 1)

// Точное решение
double trueSolution(double x, double y) {
    return exp(1 - x*x - y*y);
}

// Инициализация начального приближения (линейная интерполяция по x)
vector<vector<double>> initialize(int n, int m) {
    vector<vector<double>> u(n + 1, vector<double>(m + 1, 0.0));
    double h = 2.0 / n; // Шаг по x
    double k = 2.0 / m; // Шаг по y

    // Применение граничных условий
    for (int j = 0; j <= m; ++j) {
        double y = -1.0 + j * k;
        u[0][j] = mu1(y);
        u[n][j] = mu2(y);
    }
    for (int i = 0; i <= n; ++i) {
        double x = -1.0 + i * h;
        u[i][0] = mu3(x);
        u[i][m] = mu4(x);
    }

    // Линейная интерполяция для внутренних точек
    for (int i = 1; i < n; ++i) {
        double x = -1.0 + i * h;
        for (int j = 1; j < m; ++j) {
            double y = -1.0 + j * k;
            u[i][j] = 0.5 * (u[0][j] + u[n][j]); // Линейная интерполяция вдоль x
        }
    }

    return u;
}

// Метод верхней релаксации
vector<vector<double>> solveSOR(int n, int m, double omega, double eps, int Nmax, double& residual, double& achieved_diff) {
    vector<vector<double>> u = initialize(n, m);
    double h = 2.0 / n;
    double k = 2.0 / m;
    double h2 = h * h;
    double k2 = k * k;

    double diff = 1.0;
    int iter = 0;

    residual = 0.0;
    achieved_diff = diff;// Точность метода

    while (diff > eps && iter < Nmax) {
        diff = 0.0;
        iter++;

        for (int i = 1; i < n; ++i) {
            for (int j = 1; j < m; ++j) {
                double x = -1.0 + i * h;
                double y = -1.0 + j * k;

                double u_old = u[i][j];

                u[i][j] = (1 - omega) * u[i][j] +
                          omega / (2 * (1 / h2 + 1 / k2)) *
                          ((u[i + 1][j] + u[i - 1][j]) / h2 +
                           (u[i][j + 1] + u[i][j - 1]) / k2 -
                           f(x, y));

                diff = max(diff, abs(u[i][j] - u_old));
            }
        }
        achieved_diff = diff;
    }

    cout << "Метод SOR завершен за " << iter << " итераций." << endl;
    if (iter == Nmax) {
        cout << "Предупреждение: Достигнуто максимальное количество итераций." << endl;
    }

    // Вычисление невязки
    for (int i = 1; i < n; ++i) {
        for (int j = 1; j < m; ++j) {
            double x = -1.0 + i * h;
            double y = -1.0 + j * k;

            double res = (u[i - 1][j] - 2 * u[i][j] + u[i + 1][j]) / h2 +
                         (u[i][j - 1] - 2 * u[i][j] + u[i][j + 1]) / k2 -
                         f(x, y);

            residual = max(residual, abs(res));
        }
    }

    return u;
}

int main() {
    setlocale(LC_ALL,"Rus");
    int n = 5; // Число разбиений по x
    int m = 5; // Число разбиений по y
    double omega = 1.7; // Параметр релаксации (0 < omega < 2)
    double eps = 1e-6; // Точность
    int Nmax = 10000; // Максимальное число итераций
    double max_error, residual, achieved_diff;
    int max_error_i, max_error_j;

    vector<vector<double>> solution = solveSOR(n, m, omega, eps, Nmax, residual, achieved_diff);

    cout << fixed << setprecision(6); // Установка точности вывода

    // Вычисление максимальной погрешности
    max_error = 0.0;
    max_error_i = -1;
    max_error_j = -1;

    double h = 2.0 / n;
    double k = 2.0 / m;

    for (int i = 1; i < n; ++i) {
        for (int j = 1; j < m; ++j) {
            double x = -1.0 + i * h;
            double y = -1.0 + j * k;
            double true_val = trueSolution(x, y);

            if (!isnan(true_val)) {
                double error = abs(solution[i][j] - true_val);
                if (error > max_error) {
                    max_error = error;
                    max_error_i = i;
                    max_error_j = j;
                }
            }
        }
    }

    // Вывод погрешности и невязки
    if (!isnan(max_error)) {
        cout << "Maximum error: " << max_error << endl;
        cout << "Reached at element u[" << max_error_i << "][" << max_error_j << "]" << endl;
        cout << "Actual accuracy (diff): " << achieved_diff << endl;
        cout << "Specified accuracy: " << eps << endl;
        if (max_error <= eps) {
            cout << "Specified accuracy achieved." << endl;
        } else {
            cout << "Specified accuracy not achieved." << endl;
        }
    } else {
        cout << "Exact solution is unknown, error cannot be estimated." << endl;
        cout << "Actual accuracy (diff): " << achieved_diff << endl;
    }

    cout << "Maximum residual: " << residual << endl;

    // Вывод численного и точного решения
    cout << "\nЧисленное решение - Точное решение = Погрешность:" << endl;

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= m; ++j) {
           double x = -1.0 + i * h;
            double y = -1.0 + j * k;
            double numerical_solution = solution[i][j];
            double exact_solution = trueSolution(x, y);
             if (i > 0 && i < n && j > 0 && j < m){
               double error = abs(numerical_solution - exact_solution);
              cout << numerical_solution << " - " << exact_solution << " = " << error << "  ";
             }
             else {
                 cout << numerical_solution << " - " << exact_solution << " = NA ";
             }

        }
        cout << endl;
    }

    return 0;
}
