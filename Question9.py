import numpy as np
import matplotlib.pyplot as plt

### Adjustable parameters ###
x_s = 0.8

### Parameters ###
G_sc = 1360             # W / m2
Q = G_sc/4              # W / m2

A_out = 201.4           # W / m2
B_out = 1.45            # W / m2*C

### Functions ###
def temp(x):
    return 0.0

def S(x:float) -> float:
    S_2 = -0.477

    return 1 + S_2*0.5*(3*x**2 - 1)

def I(x:float) -> float:

    return B_out * temp(x) + A_out


def legendrePolynomials(l:int, x:list) -> list:

    def f(n:int) -> float:
        return (n*(n+1) - l*(l+1)) / ((n+2) * (n+1))
    
    if l % 2 == 0:
        legendre = np.ones(len(x))
        
        i = 0
        func_values = []
        while True:
            func = f(i)

            if func == 0:
                break

            func_values.append(func)
            value = 1
            for F in func_values:
                value *= F

            legendre += value*x**(i+2)

            i += 2

    else:
        legendre = x.copy()

        i = 1
        func_values = []
        while True:
            func = f(i)

            if func == 0:
                break

            func_values.append(func)
            value = 1
            for F in func_values:
                value *= F

            legendre += value*x**(i+2)

            i += 2

    # Determining a_0
    a_0 = 1 / legendre[-1]
    
    return a_0*legendre


# Pringing the first 6 Legendre polynomials
x = np.linspace(-1, 1, num=50)
for i in range(6):
    plt.plot(x, legendrePolynomials(i, x), label=f"$P_{i}$")
plt.xlim(-1, 1)
plt.ylim(-1.1, 1.1)
plt.legend()
plt.show()