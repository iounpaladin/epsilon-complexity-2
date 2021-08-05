import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt


def W(x):  # implementation of Weierstrass function for specific constants
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
    # find the value with which we normalize delta

    space = np.linspace(0, 0.5, 5000)
    for i in space:  # find minimal i such that delta(f, i) / R > e
        if i == 0: continue
        if delta(f, i) / R > e:
            return -np.log(i)

    return 1


def delta(f, h):
    if h > 0.5:  # cannot create the x-space if h is too large
        return 0

    x = np.linspace(0, 1, int(np.ceil(1 / h)))  # create space of samples
    y = np.array(list(map(f, x)))  # apply function to the space

    g = scipy.interpolate.interp1d(x, y)  # spline interpolation
    res = -scipy.optimize.minimize_scalar(lambda q: -abs(g(q) - f(q)), bounds=[0, 1], method='bounded').fun
    # find maximal error (negate the minimum)

    return res


def run_complexity(f):
    x = []
    y = []
    for e in [
        0.5,
        0.1,
        0.05,
        0.01,
        0.005,
        0.001
    ]:
        x.append(np.log(e))  # for several values of epsilon, record log(epsilon)
        y.append(complexity(f, e))  # and S_e(f)

    coeff = np.polynomial.polynomial.polyfit(x, y, 1)  # best-fit line

    return x, y, coeff


if __name__ == "__main__":
    def fmt(a):
        return f"{round(a[0], 4)} + {round(a[1], 4)} log ε"  # formatting for graph title


    x = np.linspace(0, 1, num=200)
    y = W(x)
    plt.plot(x, y, '-')
    plt.title('W')  # graph Weierstrass function (just for fun)
    plt.show()
    for func, name in [
        (lambda x: np.sin(2 * np.pi * x), 'sin'),
        (lambda x: np.cos(2 * np.pi * x), 'cos'),
        (lambda x: x ** (1 / 2), 'sqrt'),
        (lambda x: x ** (1 / 3), 'cbrt'),
        (W, 'W'),
    ]:
        x, y, coeff = run_complexity(func)  # for each function, plot the recorded values and best fit line

        plt.plot(x, y, 'o', x, np.polynomial.polynomial.polyval(x, coeff))
        plt.title(name + ' ' + fmt(coeff))
        plt.show()
