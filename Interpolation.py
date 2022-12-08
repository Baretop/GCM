import math as m
import numpy as np


def limiter(y, yArray):
    max = np.max(yArray)
    if y >= max:
        return max

    min = np.min(yArray)
    if y <= min:
        return min

    return y


def linearInterpolation(x, xArray, h, yArray, useLimiter=True):
    index = m.floor((x - xArray[0]) / h)

    # Вылезли справа. Шаблон начинается на одну точку левее
    if x >= xArray[-1]:
        index -= 1

    x0 = xArray[index]
    x1 = xArray[index + 1]

    return yArray[index] * (x1 - x) / h + yArray[index + 1] * (x - x0) / h


def quadraticInterpolation(x, xArray, h, yArray, useLimiter=True):
    index = m.floor((x - xArray[0]) / h)

    # Вылезли справа. Шаблон начинается на одну точку левее
    if x >= xArray[-1]:
        index -= 1
    # Вылезли слева. Шаблон начинается на одну точку правее
    if x < xArray[1]:
        index += 1

    x0 = xArray[index - 1]
    x1 = xArray[index]
    x2 = xArray[index + 1]

    result = yArray[index - 1] * (x - x1) * (x - x2) / (2 * h ** 2) + yArray[index] * (x - x0) * (x - x2) / (-h ** 2) + \
             yArray[index + 1] * (x - x0) * (x - x1) / (2 * h ** 2)

    if useLimiter:
        return limiter(result, yArray[index - 1:index + 2])
    return result


# Кубическая интерполяция, метод Лагранжа
def сubicInterpolation(x, xArray, h, yArray, useLimiter=True):
    index = m.floor((x - xArray[0]) / h)

    # Вылезли справа. Шаблон начинается на одну точку левее
    if x >= xArray[-2]:
        index -= 1
    # Вылезли слева. Шаблон начинается на одну точку правее
    if x < xArray[1]:
        index += 1

    x0 = xArray[index - 1]
    x1 = xArray[index]
    x2 = xArray[index + 1]
    x3 = xArray[index + 2]

    result = yArray[index - 1] * (x - x1) * (x - x2) * (x - x3) / (-6 * h ** 3) + yArray[index] * (x - x0) * (
            x - x2) * (x - x3) / (2 * h ** 3) + yArray[index + 1] * (x - x0) * (x - x1) * (x - x3) / (-2 * h ** 3) + \
             yArray[index + 2] * (x - x0) * (x - x1) * (x - x2) / (6 * h ** 3)

    if useLimiter:
        return limiter(result, yArray[index - 1:index + 3])
    return result

def wrapperInterpolation(x, xArray, h, yArray, orderInterpolation, useLimiter=True):
    if orderInterpolation == 1:
        return linearInterpolation(x, xArray, h, yArray, useLimiter)
    if orderInterpolation == 2:
        return quadraticInterpolation(x, xArray, h, yArray, useLimiter)
    if orderInterpolation == 3:
        return сubicInterpolation(x, xArray, h, yArray, useLimiter)
