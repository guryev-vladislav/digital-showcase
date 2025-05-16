//
// Created by gurye on 16.05.2025.
//

#ifndef SPLINECALCULATOR_H
#define SPLINECALCULATOR_H

#pragma once
#include <vector>
#include <cmath>
#include <functional> // Для std::function

class SplineCalculator {
public:
    // Используем std::function для гибкости (можно передавать лямбда-функции, указатели на функции и т.д.)
    using FunctionType = std::function<double(double)>;

    SplineCalculator(const std::vector<double>& x, const std::vector<double>& y,
                     FunctionType f_prime, FunctionType f_double_prime);

    std::vector<double> solve();
    std::vector<double> count_a();
    std::vector<double> count_h();
    std::vector<double> count_F();
    std::vector<double> count_S(const std::vector<double>& x_un);
    std::vector<double> count_pogr(const std::vector<double>& x_un, const std::vector<double>& y_un);
    std::vector<double> count_pogr_pr1(const std::vector<double>& x_un, const std::vector<double>& y_un);
    std::vector<double> count_pogr_pr2(const std::vector<double>& x_un, const std::vector<double>& y_un);
    std::vector<std::pair<double, int>> findMax(const std::vector<double>& arr);
    double Spline_N_2(double t);

private:
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> a;
    std::vector<double> h;
    std::vector<double> F;
    std::vector<double> S;
    std::vector<double> S_pr1;
    std::vector<double> S_pr2;
    std::vector<double> m;
    int n;
    FunctionType F_prime;
    FunctionType F_double_prime;
};



#endif //SPLINECALCULATOR_H
