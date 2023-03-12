import numpy as np
import math

def diagonal_preobl(matrix):
    n = matrix.shape[0]
    for i in range(n):
        sum = 0
        for j in range(n):
            if j != i:
                sum += abs(matrix[i, j])
        if sum > abs(matrix[i, i]):
            return False
    return True

def matrixWithDiagonalDominance(n):                     #конструирование матрицы с диаг преобл.
    matrix = 110*np.random.random_sample((n,n)) - 30

    for i in range(n):
        sum = 0.0
        for j in range(n):
            sum += abs(matrix[i, j])
        matrix[i, i] = sum

    return matrix


def Decomposition(a, b):              # перевод уравнения в вид: x = Bx + c
    a = a.copy()
    n = np.size(a, 0)

    l = np.zeros((n, n), dtype='float64')
    invD = np.zeros((n, n), dtype='float64')
    r = np.zeros((n, n), dtype='float64')

    for i in range(n):
        invD[i, i] = 1.0 / a[i, i]

    for i in range(n):
        for j in range(0, i):
            l[i, j] = a[i, j]

    for i in range(n):
        for j in range(i + 1, n):
            r[i, j] = a[i, j]

    return (np.dot(-invD, l + r), np.dot(invD, b))

def Jacobi(a, b, eps = 0.000001):
    (B, c) = Decomposition(a, b)
    q = np.linalg.norm(B, 1)
    x = c.copy()                #начальное значение: x = с
    prevX = x.copy()
    x = np.dot(B, prevX) + c
    aprioriEstimate = math.ceil(math.log( eps * np.linalg.norm(x - prevX, 1), q ))

    threshold = (1.0 - q) / q * eps

    index = 0
    while np.linalg.norm(x - prevX, 1) > threshold:
        index += 1
        prevX = x.copy()
        x = np.dot(B, x) + c

    return (x, aprioriEstimate, index)


def Seidel(a, b, eps = 0.000001):
    n = np.size(a, 0)
    (B, c) = Decomposition(a, b)
    q = np.linalg.norm(B, 1)
    x = c.copy()
    prevX = x.copy()
    x = np.dot(B, prevX) + c
    aprioriEstimate = math.ceil(math.log(eps * np.linalg.norm(x - prevX, 1), q))

    threshold = (1.0 - q) / q * eps

    index = 0
    while np.linalg.norm(x - prevX, 1) > threshold:
        index += 1
        prevX = x.copy()
        for i in range(n):
            newValue = c[i, 0]
            for j in range(n):
                newValue += B[i, j] * x[j, 0]
            x[i, 0] = newValue

    return (x, aprioriEstimate, index)


size_of_matrix = 5
b = 110*np.random.random_sample((size_of_matrix,1)) - 30
#b = np.array([[1],[2],[2]])
# matrix = 110*np.random.random_sample((size_of_matrix,size_of_matrix)) - 30
# while np.linalg.det(matrix) == 0:
#     matrix = 110*np.random.random_sample((size_of_matrix,size_of_matrix)) - 30
#matrix = np.array([[0,6,10],[-3,-7,2],[5,-1,5]])
matrix = matrixWithDiagonalDominance(size_of_matrix)

print(matrix)

print("Метод Якоби")
tmp = Jacobi(matrix, b)
print('решение')
print(tmp[0])
print('Априорная оценка')
print(tmp[1])
print('Апостериорная оценка')
print(tmp[2])

print("===============")

print("Метод Зейделя")
tmp = Seidel(matrix, b)
print('решение')
print(tmp[0])
print('Априорная оценка')
print(tmp[1])
print('Апостериорная оценка')
print(tmp[2])