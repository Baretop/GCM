import numpy as np
import math as m
import matplotlib.pyplot as plt
from Interpolation import *
from scipy.interpolate import CubicSpline
import csv

# Условия области, в которой интегрируем
xBegin, xEnd = 0, 150
tBegin, tEnd = 0, 0.2
leftBorder = xBegin + 0.1 * (xEnd - xBegin)  # задаем левую границу нач. импульса
rightBorder = xBegin + 0.3 * (xEnd - xBegin)  # задаем правую границу нач. импульса
courant = 0.5
Lambda, mu, rho = 70000, 10000, 1


# # Условие на функцию при t = 0
# def initialCondition(x):
#     if leftBorder < x < rightBorder:
#         return m.exp(-4 * (2 * x - (leftBorder + rightBorder)) ** 2 / (
#                 (rightBorder - leftBorder) ** 2 - (2 * x - (leftBorder + rightBorder)) ** 2))
#     return 0

def jump(x):
    if leftBorder < x < rightBorder:
        return 1
    return 0


def sinSquared(x):
    arg = x / xEnd
    if 0.1 < arg < 0.3:
        return m.sin(5 * m.pi * (arg + 0.1)) ** 2
    return 0


def analSol(t, x, initialCondition):
    return 90000 * initialCondition(x - 300 * t)


# Параметры, необходимые для работы с инвариантами Римана
omega = np.array([[-m.sqrt((Lambda + 2 * mu) * rho), 1], [m.sqrt((Lambda + 2 * mu) * rho), 1]])
omegaInv = 1 / 2 / m.sqrt((Lambda + 2 * mu) * rho) * np.array(
    [[-1, 1], [m.sqrt((Lambda + 2 * mu) * rho), m.sqrt((Lambda + 2 * mu) * rho)]])
eigenvalues = [m.sqrt((Lambda + 2 * mu) / rho), -m.sqrt((Lambda + 2 * mu) / rho)]


# Расчёт по СХМ
def calc(initialCondition, I, orderInterpolation, useLimiter, useCubicSpline=False):
    N = 1 + int(300 * (tEnd - tBegin) * (I - 1) / courant / (xEnd - xBegin))  # число точек сетки по времени
    tau = (tEnd - tBegin) / (N - 1)

    xArray = np.linspace(xBegin, xEnd, I)
    h = xArray[1] - xArray[0]

    # Массив искомых переменных:
    u = np.zeros((N, I, 2))
    # Массив искомых переменных в виде инвариантов Римана (храним только текущее и пред. зн-е)
    w = np.zeros((2, I, 2))

    # Задаем начальные условия
    for i in range(I):
        u[0, i, :] = initialCondition(xBegin + h * i) * np.array([eigenvalues[1], Lambda + 2 * mu])

    # Переводим начальное значение в инварианты Римана
    w[0] = np.einsum("ij,kj->ki", omega, u[0])

    # Решаем СХМ
    for n in range(1, N):
        if not useCubicSpline:
            for i in range(I):
                # Применяем к каждой компоненте перенос значений (если это возможно, см. гран. усл-я)
                for k in range(len(eigenvalues)):
                    # Вычисляем координату, в которой нужно проинтерполировать функцию
                    x = (i * h + xArray[0]) - eigenvalues[k] * tau
                    if x > xArray[0] and x < xArray[-1]:
                        # Интерполируем
                        w[1, i, k] = wrapperInterpolation(x, xArray, h, w[0, :, k], orderInterpolation, useLimiter)
                    else:
                        # Сделано в тупую, чтобы не париться сейчас с граничными условиями
                        w[1, i, k] = 0
        else:
            cubicSplines = [CubicSpline(xArray, w[0, :, 0], bc_type='natural'),
                            CubicSpline(xArray, w[0, :, 1], bc_type='natural')]
            for i in range(I):
                # Применяем к каждой компоненте перенос значений (если это возможно, см. гран. усл-я)
                for k in range(len(eigenvalues)):
                    # Вычисляем координату, в которой нужно проинтерполировать функцию
                    x = (i * h + xArray[0]) - eigenvalues[k] * tau
                    if x > xArray[0] and x < xArray[-1]:
                        # Интерполируем
                        w[1, i, k] = cubicSplines[k](x)
                        if useLimiter:
                            index = m.floor((x - xArray[0]) / h)
                            w[1, i, k] = limiter(w[1, i, k], w[0, index:(index + 2), k])
                    else:
                        w[1, i, k] = 0

        # Возвращаемся к исходным переменным:
        u[n] = np.einsum("ij,kj->ki", omegaInv, w[1])

        w[0] = w[1]
        w[1] = np.zeros((I, 2))

    # Аналитическое решение
    analSigma = np.array([analSol(tEnd, x, initialCondition) for x in xArray])

    delta = np.abs(analSigma - u[-1, :, 1])
    # Максимум модуля ошибки (C-норма)
    error1 = delta.max()
    # Среднее арифметическое модулей ошибок (L1 - норма)
    error2 = np.sum(delta) / I
    # Вычисляю евклидову норму (L2 - норма)
    error3 = m.sqrt(np.sum(np.square(delta)) / I)
    return np.array([h, error1, error2, error3]), xArray, u[-1, :, 1]


def vizualize(file_writer, firstName, secondName, IArray, initialCondition, limit, order, useSpline):
    xArrayForVizualize = np.linspace(xBegin, xEnd, 501)
    analSigma = np.array([analSol(tEnd, x, initialCondition) for x in xArrayForVizualize])

    result = np.zeros((4, len(IArray)))
    for i in range(len(IArray)):
        result[:, i], xArray, yArray = calc(initialCondition, IArray[i], order, limit, useSpline)

        fig, ax = plt.subplots()
        fig.set_dpi(300)
        ax.plot(xArrayForVizualize, analSigma, xArray, yArray, linewidth=1)
        ax.grid()
        fig.savefig("results/images/" + ("jump" if initialCondition is jump else "sin^2") + "/" + firstName + "/" + str(
            IArray[i]) + (", limit" if limit else "") + ".png")
        plt.close()

    result = np.log(result)

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(12, 5)
    fig.set_dpi(300)
    fig.suptitle('Графики сходимости, С, L1, L2 нормы')
    z = []
    for i in range(3):
        z.append(np.polyfit(result[0], result[i + 1], 1))
        p = np.poly1d(z[-1])

        ax[i].plot(result[0], result[i + 1], marker=".")
        ax[i].plot(result[0], p(result[0]))
        ax[i].grid()
    fig.savefig("results/images/" + (
        "jump" if initialCondition is jump else "sin^2") + "/" + firstName + "/plot for calculate order" + (
                    ", limit" if limit else "") + ".png")
    plt.close()

    file_writer.writerow(
        [secondName, "+" if limit else "-", ("jump" if initialCondition is jump else "sin^2"), f'{z[0][0]:.3f}',
         f'{z[1][0]:.3f}', f'{z[2][0]:.3f}'])


IArray = [51, 101, 201, 401, 801, 1601]
orderInterpolation = [1, 2, 3]
useLimiter = False, True

with open("results/tables/order of approximation.csv", mode="w", encoding='utf-8') as w_file:
    file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
    file_writer.writerow(["теор. порядок", "лимитер", "фронт", "C", "L1", "L2"])

    for initialCondition in [jump, sinSquared]:
        for limit in useLimiter:
            for order in orderInterpolation:
                vizualize(file_writer, str(order), str(order), IArray, initialCondition, limit, order, False)
            vizualize(file_writer, "cubicSpline", "3 сплайн", IArray, initialCondition, limit, order, True)
