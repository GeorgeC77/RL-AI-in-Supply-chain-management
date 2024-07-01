import numpy as np
from scipy.stats import norm

L = 3
sigma = 2

p = 9
h = 1
u = 4
w = 6
mu = 0
m = 1.5

cap_Coeff = norm.pdf(norm.ppf((w - u) / w))
inv_Coeff = norm.pdf(norm.ppf(p / (p + h)))

print(u*m*cap_Coeff/((h+p)*inv_Coeff+u*m*cap_Coeff))

alpha = np.arange(-0.99, 1, 0.01)

for rho in [-0.9, -0.5, 0, 0.5, 0.9]:
    Var_O1 = (2 * (1 - alpha) * rho ** L * (rho + 1) * (rho - alpha) - 2 * rho ** (2 * L) * (rho - alpha) ** 2 - (
                1 - alpha) * (rho + 1) * (1 - rho * alpha)) / (-1 - alpha) / (rho - 1) ** 2 / (rho + 1) / (
                         1 - rho * alpha) * sigma ** 2

    Var_NS1 = ((L * (1 - rho ** 2) + rho * (1 - rho ** L) * (rho ** (L + 1) - rho - 2)) / (1 - rho) ** 2 / (
                1 - rho ** 2) + (rho ** L - 1) ** 2 * alpha ** 2 / (rho - 1) ** 2 / (1 + alpha) / (
                           1 - alpha)) * sigma ** 2

    Var_NS2 = ((L*(1-rho**2) + rho**(L+1) * (1-rho**L) * (rho**(L+1) + rho**(2*L+1) - 2*rho -2)) / (1-rho)**2 / (1-rho**2) + (2*(rho**L-1)*alpha*(((1-alpha**L)/(1-alpha)-rho**(L+1)*
            ((rho*alpha)**L-1)/(rho*alpha-1))+(alpha**(2*L)-1)*(rho**L-1)**2*alpha**2/(alpha**2-1)+((1-alpha**L-rho**L*(rho**L-alpha**L))**2*alpha**2)/(1+alpha)/(1-alpha)))/(1-rho)**2)*sigma**2

    # J = (h + p) * np.sqrt(Var_NS1) * inv_Coeff + mu * u + u * m * cap_Coeff * np.sqrt(Var_O1)

    # retailer
    J = (h + p) * np.sqrt(Var_NS1) * inv_Coeff



    alpha_prime = np.arange(-0.99, 1, 0.01)[np.argmin(J)]
    Var_O1_prime = (2 * (1 - alpha_prime) * rho ** L * (rho + 1) * (rho - alpha_prime) - 2 * rho ** (2 * L) * (rho - alpha_prime) ** 2 - (
                1 - alpha_prime) * (rho + 1) * (1 - rho * alpha_prime)) / (-1 - alpha_prime) / (rho - 1) ** 2 / (rho + 1) / (
                         1 - rho * alpha_prime) * sigma ** 2
    k_prime = mu + cap_Coeff * np.sqrt(Var_O1_prime)

    # manufacturer
    J_2 = (h + p) * np.sqrt(Var_NS2) * inv_Coeff

    J_2_prime = np.min(J_2)


    print(rho, alpha_prime, k_prime, np.min(J), np.sqrt(Var_O1_prime))

    print(rho, )