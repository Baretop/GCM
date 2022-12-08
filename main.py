import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib import animation
from Interpolation import *
from scipy.interpolate import CubicSpline

# Условия области, в которой интегрируем
xBegin, xEnd = 0, 300
tBegin, tEnd = 0, 0.2
leftBorder = xBegin + 0.2 * (xEnd - xBegin)  # задаем левую границу нач. импульса
rightBorder = xBegin + 0.3 * (xEnd - xBegin)  # задаем правую границу нач. импульса
courant = 0.5
Lambda, mu, rho = 70000, 10000, 1


# Условие на функцию при t = 0
def initialCondition(x):
    if leftBorder < x < rightBorder:
        return m.exp(-4 * (2 * x - (leftBorder + rightBorder)) ** 2 / (
                (rightBorder - leftBorder) ** 2 - (2 * x - (leftBorder + rightBorder)) ** 2))
    return 0


def analSol(t, x):
    return 90000 * initialCondition(x - 300 * t)


# Параметры, необходимые для работы с инвариантами Римана
omega = np.array([[-m.sqrt((Lambda + 2 * mu) * rho), 1], [m.sqrt((Lambda + 2 * mu) * rho), 1]])
omegaInv = 1 / 2 / m.sqrt((Lambda + 2 * mu) * rho) * np.array(
    [[-1, 1], [m.sqrt((Lambda + 2 * mu) * rho), m.sqrt((Lambda + 2 * mu) * rho)]])
eigenvalues = [m.sqrt((Lambda + 2 * mu) / rho), -m.sqrt((Lambda + 2 * mu) / rho)]


# Расчёт по СХМ
def calc(I, orderInterpolation, useLimiter, useCubicSpline=False):
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
    analSigma = np.array([analSol(tEnd, x) for x in xArray])

    # Вычисляю евклидову норму. Может быть здесь ошибка?
    error = m.sqrt(np.sum(np.square(analSigma - u[-1, :, 1]))) / I
    return np.array([h, error])


# ЭТОТ БЛОК ДЛЯ ТОГО ЧТОБЫ ПОСМОТРЕТЬ РЕЗУЛЬТАТЫ РАСЧЁТА ДЛЯ КОНКРЕТНОГО МЕТОДА, С ГРАФИКОМ
# IArray = [101, 201, 401, 801, 1601, 3201]
# orderInterpolation = [1, 2, 3]
#
# result = np.zeros((2, len(IArray)))
# for i in range(len(IArray)):
#     result[:, i] = calc(IArray[i], True, 3)
# print(result[0])
# print(result[1])
#
# result = np.log(result)
# z = np.polyfit(result[0], result[1], 1)
# print(z)
# p = np.poly1d(z)
#
#
# fig = plt.figure()
# ax = plt.axes(xlim=(m.log(300 / (IArray[-1] - 1)), m.log(300 / (IArray[0] - 1))), ylim=(np.amin(result[1]), np.amax(result[1])))
# ax.plot(result[0], result[1], result[0], p(result[0]))
# ax.grid()
#
# plt.show()

IArray = [101, 201, 401, 801, 1601, 3201]
orderInterpolation = [1, 2, 3]
useLimiter = False, True

print("ЧИСЛО КУРАНТА: " + str(courant))

for order in orderInterpolation:
    for limit in useLimiter:
        result = np.zeros((2, len(IArray)))
        for i in range(len(IArray)):
            result[:, i] = calc(IArray[i], order, limit, False)

        result = np.log(result)

        z = np.polyfit(result[0], result[1], 1)

        print("Order: " + str(order))
        print("Use limiter: " + str(limit))
        print("p: " + str(z[0]))
        print("====================")

for limit in useLimiter:
    result = np.zeros((2, len(IArray)))
    for i in range(len(IArray)):
        result[:, i] = calc(IArray[i], -1, limit, True)

    result = np.log(result)

    z = np.polyfit(result[0], result[1], 1)

    print("CubicSpline")
    print("Use limiter: " + str(limit))
    print("p: " + str(z[0]))
    print("====================")
