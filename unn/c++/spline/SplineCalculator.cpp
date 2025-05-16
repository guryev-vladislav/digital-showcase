#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip> // Для форматирования вывода
#include <limits> // Для проверки ввода

class SplineCalculator {
public:
    SplineCalculator(const std::vector<double>& x, const std::vector<double>& f) : x_un(x), f_un(f) {
        n = x.size() - 1;
        a.resize(n + 1);
        b.resize(n);
        c.resize(n + 1);
        d.resize(n);
        h.resize(n); // Инициализация вектора h

        count_a();
        count_h(); // Вычисление h
        count_c();
        count_b();
        count_d();
    }

    std::vector<double> count_S(const std::vector<double>& x_new) {
        std::vector<double> result;
        for (double xi : x_new) {
            double s_val = 0.0;
            for (int i = 0; i < n; ++i) {
                if (xi >= x_un[i] && xi <= x_un[i + 1]) {
                    s_val = a[i] + b[i] * (xi - x_un[i]) + c[i] / 2.0 * pow(xi - x_un[i], 2) + d[i] / 6.0 * pow(xi - x_un[i], 3);
                    break;
                }
            }
            result.push_back(s_val);
        }
        return result;
    }

    std::vector<double> count_pogr(const std::vector<double>& s_x, const std::vector<double>& f_x) {
        std::vector<double> result;
        for (int i = 0; i < x_un.size(); ++i) {
            result.push_back(fabs(f_x[i] - s_x[i]));
        }
        return result;
    }

    std::pair<double, double> findMax(const std::vector<double>& arr) {
        double maxVal = 0.0;
        double maxX = 0.0;
        for (int i = 0; i < arr.size(); ++i) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxX = x_un[i];
            }
        }
        return {maxVal, maxX};
    }

    // Функция для записи данных в CSV-файл
    void writeDataToCSV(const std::string& filename, const std::vector<double>& x, const std::vector<double>& f_x, const std::vector<double>& s_x, const std::vector<double>& pogr) {
        std::ofstream outputFile(filename);
        if (outputFile.is_open()) {
            outputFile << "x,f_x,s_x,pogr\n"; // Header
            for (size_t i = 0; i < x.size(); ++i) {
                outputFile << std::fixed << std::setprecision(10) << x[i] << "," << f_x[i] << "," << s_x[i] << "," << pogr[i] << "\n";
            }
            outputFile.close();
            std::cout << "Data saved to " << filename << std::endl;
        } else {
            std::cerr << "Could not open the file for writing.\n";
        }
    }

private:
    int n;
    std::vector<double> x_un;
    std::vector<double> f_un;
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;
    std::vector<double> d;
    std::vector<double> h; // Добавлено объявление h

    void count_a() {
        for (int i = 0; i <= n; ++i) {
            a[i] = f_un[i];
        }
    }

    void count_h() { // Функция для вычисления h
        for (int i = 0; i < n; ++i) {
            h[i] = x_un[i + 1] - x_un[i];
        }
    }

    void count_c() {
        std::vector<double> alpha(n + 1);
        std::vector<double> beta(n + 1);

        alpha[0] = 0;
        beta[0] = 0;

        for (int i = 1; i < n; ++i) {
            alpha[i] = -h[i] / (h[i - 1] + h[i]);
            beta[i] = (3 * ((f_un[i + 1] - f_un[i]) / h[i] - (f_un[i] - f_un[i - 1]) / h[i - 1])) / (h[i - 1] + h[i]);
        }

        c[0] = 0;
        c[n] = 0;

        for (int i = 1; i < n; ++i) {
            c[i] = (beta[i] - alpha[i] * c[i - 1]) / (alpha[i] + 1);
        }

        for (int i = n - 1; i >= 0; --i) {
            c[i] = alpha[i + 1] * c[i + 1] + beta[i + 1];
        }
    }

    void count_b() {
        for (int i = 0; i < n; ++i) {
            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] / 3.0 * c[i] - h[i] / 6.0 * c[i + 1];
        }
    }

    void count_d() {
        for (int i = 0; i < n; ++i) {
            d[i] = (c[i + 1] - c[i]) / h[i];
        }
    }
};

// Пример использования
int main() {
    // --- Start of Input Data ---
    std::vector<double> x_values; // X-coordinates of the data points
    std::vector<double> f_values; // Y-coordinates (function values) of the data points
    double x_start, x_end, x_step; // Range and step for spline evaluation

    // Input number of data points
    int num_points;
    std::cout << "Enter the number of data points: ";
    std::cin >> num_points;

    // Input x and f(x) values
    std::cout << "Enter x and f(x) values for each point:\n";
    for (int i = 0; i < num_points; ++i) {
        double x, f;
        std::cout << "x[" << i << "]: ";
        std::cin >> x;
        x_values.push_back(x);
        std::cout << "f(x)[" << i << "]: ";
        std::cin >> f;
        f_values.push_back(f);
    }

    // Input range for spline evaluation
    std::cout << "Enter the start x value for spline evaluation: ";
    std::cin >> x_start;
    std::cout << "Enter the end x value for spline evaluation: ";
    std::cin >> x_end;
    std::cout << "Enter the step size for spline evaluation: ";
    std::cin >> x_step;

    // --- End of Input Data ---

    // Generate x values for spline evaluation
    std::vector<double> x_spline;
    for (double i = x_start; i <= x_end; i += x_step) {
        x_spline.push_back(i);
    }

    SplineCalculator spline(x_values, f_values);
    std::vector<double> s_values = spline.count_S(x_spline);

    // Вычисление погрешности (на исходной сетке)
    std::vector<double> s_original = spline.count_S(x_values);
    std::vector<double> pogr_values = spline.count_pogr(s_original, f_values);

    // Запись данных в CSV
    spline.writeDataToCSV("spline_data.csv", x_spline, std::vector<double>(x_spline.size()), s_values, std::vector<double>(x_spline.size())); // Заполняем нулями f_x и pogr для x_spline
    spline.writeDataToCSV("original_data.csv", x_values, f_values, s_original, pogr_values);

    std::pair<double, double> max_pogr = spline.findMax(pogr_values);
    std::cout << "Maximum error: " << max_pogr.first << " at x = " << max_pogr.second << std::endl;

    return 0;
}