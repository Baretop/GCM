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


IArray = [26, 51, 101, 201, 401, 801, 1601, 3201]

result = np.zeros((4, len(IArray)))
for i in range(len(IArray)):
    result[:, i], xArray, yArray = calc(sinSquared, IArray[i], 1, False, False)

result = np.log(result)

fig, ax = plt.subplots()
fig.set_dpi(300)
fig.suptitle('Зависимость фактического порядка аппроксимации \n от шага на примере схемы 1 порядка')
ax.set_xlabel("ln(h)")
ax.set_ylabel("ln(Δ)")

z1 = np.polyfit(result[0, :2], result[3, :2], 1)
z2 = np.polyfit(result[0, -1:-3:-1], result[3, -1:-3:-1], 1)
p1 = np.poly1d(z1)
p2 = np.poly1d(z2)

ax.plot(result[0], result[3], marker=".")
ax.plot(result[0, :3], p1(result[0, :3]))
ax.text(0.5, 9.4, f'{z1[0]:.3f}')
ax.plot(result[0, -1:-4:-1], p2(result[0, -1:-4:-1]))
ax.text(-2.9, 6.7, f'{z2[0]:.3f}')
ax.grid()
fig.savefig("results/images/sin^2/1/actual order of approximation.png")
plt.close()

