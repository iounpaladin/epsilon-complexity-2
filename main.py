import numpy as np
import scipy as sp
import scipy.optimize
import scipy.interpolate
from scipy.optimize import LinearConstraint, NonlinearConstraint
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt


def W(x):
    a = 0.85
    b = 7

    return sum(
        [a ** n * np.cos(b ** n * np.pi * x) for n in range(0, 30)]
    )


def complexity(f, e):
    """
    :param f: the function
    :param e: epsilon
    :return: the function's epsilon-complexity
    """

    R = -scipy.optimize.minimize_scalar(lambda x: -np.abs(f(x)), bounds=[0, 1], method='bounded').fun
    m = 0
    M = 0.5
    h = 0.25

    space = np.linspace(0, 0.5, 5000)
    for i in space:
        if i == 0: continue
        if delta(f, i) / R > e:
            return -np.log(i)

    return 1
    # for i in range(100):  # steps of binary search
    #     g = delta(f, h) / R
    #     if g > e:  # lower h if delta > e
    #         M = h
    #     else:  # otherwise, raise h (because delta is increasing in h, this always works)
    #         m = h

        # h = (m + M) / 2
    # return -np.log(h)


def delta(f, h, plot=False):
    if h > 0.5:
        return 0

    x = np.linspace(0, 1, int(np.ceil(1 / h)))
    y = np.array(list(map(f, x)))

    g = scipy.interpolate.interp1d(x, y)
    res = -scipy.optimize.minimize_scalar(lambda q: -abs(g(q) - f(q)), bounds=[0, 1], method='bounded').fun

    if plot:
        f2 = scipy.interpolate.interp1d(x, y, kind='cubic')
        plt.plot(x, y, 'o', x, g(x), '-', x, f2(x), '--', x, np.array(list(map(f, x))), '-')
        plt.legend(['data', 'linear', 'cubic', 'original'], loc='best')
        plt.show()

    return res


def run_complexity(f):
    x = []
    y = []
    for e in [
        # 2,
        # 1.5,
        # 1.75
        0.5,
        0.1,
        0.05,
        0.01,
        0.005,
        0.001
    ]:
        x.append(np.log(e))
        y.append(complexity(f, e))

    coeff = np.polynomial.polynomial.polyfit(x, y, 1)

    return x, y, coeff


if __name__ == "__main__":
    def fmt(a):
        return f"{round(a[0], 4)} + {round(a[1], 4)} log ε"


    x = np.linspace(0, 1, num=200)
    y = W(x)
    plt.plot(x, y, '-')
    plt.title('W')
    plt.show()
    for func, name in [
        (lambda x: np.sin(2 * np.pi * x), 'sin'),
        (lambda x: np.cos(2 * np.pi * x), 'cos'),
        (lambda x: x ** (1 / 2), 'sqrt'),
        (lambda x: x ** (1 / 3), 'cbrt'),
        (W, 'W'),
        # (lambda x: np.sin(1 / (x + 0.0001)), 'sin(1/x)'),
        # (lambda x: np.cos(1 / (x + 0.0001)), 'cos(1/x)'),
    ]:
        x, y, coeff = run_complexity(func)

        plt.plot(x, y, 'o', x, np.polynomial.polynomial.polyval(x, coeff))
        plt.title(name + ' ' + fmt(coeff))
        plt.show()
