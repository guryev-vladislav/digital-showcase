#pragma once

double integral_trapezoid(int n, double a, double b, double(*)(double));
double integral_simpson(int n, double a, double b, double(*)(double));
double integral_Gauss(int n, double a, double b, double(*)(double));
double Ci(int n, int i);
double ti(int n, int i);
double P0(double x);
double P1(double x);
double P2(double x);
double P3(double x);
double P4(double x);
double P5(double x);
double P(int n, double x);
