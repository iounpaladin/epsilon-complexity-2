import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate


@np.vectorize
def W(x):  # implementation of Weierstrass function for specific constants
    a = 0.85
    b = 7

    return sum(
        [a ** n * np.cos(b ** n * np.pi * x) for n in range(0, 30)]
    )


def complexity(func: callable, n: int):
    """
    Epsilon-complexity generator on a discrete lattice
    :param func: function to calculate the complexity of
    :param n: number of lattice points
    :return: tuple of points (S, e) for several values of S
    """

    for S in [1 / 2, 1 / 10, 1 / 20, 1 / 25, 1 / 40]:  # for each value of S,
        original_space = np.linspace(0, 1, n)  # take the n datapoints (n = 10,000 here)
        modified_space = np.linspace(0, 1, int(S * n))  # and the [Sn] datapoints

        recovery = scipy.interpolate.interp1d(modified_space, func(modified_space))
        # create the recovery mapping (spline interpolation) over the [Sn] datapoints

        e = max(map(lambda x: abs(recovery(x) - func(x)), original_space))
        # find the largest difference between the recovery and the original function over the n datapoints

        yield np.log(S), np.log(e)  # return log(S) vs log(e)


def graph_complexity(func: callable, func_name: str, n: int = 10000):
    """
    Graphs the best-fit line for the epsilon-complexity
    :param func: function to find the complexity of
    :param func_name name of the function for the graph title
    :param n: number of lattice points
    """

    data = list(complexity(func, n))  # get data points from complexity function
    x = list(map(lambda x: x[0], data))  # extract into x and y components
    y = list(map(lambda y: y[1], data))
    coeff = np.polynomial.polynomial.polyfit(x, y, 1)  # best-fit line

    plt.plot(x, y, 'o', x, np.polynomial.polynomial.polyval(x, coeff))  # plot
    plt.title(f'{func_name}: log ε ≈ {round(coeff[0], 2)} + {round(coeff[1], 2)} log S')
    plt.show()


graph_complexity(np.sin, "sin")
graph_complexity(np.cos, "cos")
graph_complexity(np.sqrt, "sqrt")
graph_complexity(np.cbrt, "cbrt")
graph_complexity(W, "W")
