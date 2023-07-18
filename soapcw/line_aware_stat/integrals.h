#ifndef INTEGRALS_H
#define INTEGRALS_H

double integral_signal_1det(double g, double logfrac, int k, int N, double pv);
double ch2_noise_1det(double g, int k, int N);
double integral_signal_2det(double g1, double g2, double logfrac, int k, int N, double pv);
double integral_line_2det(double g1, double g2, double logfrac, int k, int N, double pv);
double ch2_noise_2det(double g1, double g2, int k, int N);

#endif
