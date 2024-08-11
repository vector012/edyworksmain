import math as m
import numpy as np


def row(x, eps, N):
    summ = 0
    k = 1
    an = ((-1) ** (k - 1) / k) * (x ** k)
    while an > eps or k < N:
        summ = summ + an
        k = k + 1
        an = ((-1) ** (k - 1) / k) * (x ** k)
    k = k - 1
    an = ((-1) ** (k - 1) / k) * (x ** k)
    print(f"{k - 1:4}{x:12.5e}{m.log(1 + x):12.5e}{summ:12.5e}{an:12.5e}")


print('Рассчёт суммы ряда sum{1->N}:((-1)**(k-1)/k)*(x**k)=ln(1+x), |x|<1. Введите удовлетворяющие данные:')
left = float(input(' Левая граница = '))
right = float(input(' Правая = '))
amount = int(input(' Число точек = '))
eps = float(input(' Точность = '))
N = int(input(' Верхний преде суммы = '))
step = (right - left) / amount

print('result')
print(f"{' ' * 3}{'i'}{' ' * 6}{'x'}{' ' * 8}{'ln(1+x)'}{' ' * 7}{'sum'}{' ' * 8}{'an'}")
for x in np.arange(left, right, step):
    row(x, eps, N)

if left < -1 or right > 1 or left > right or step == 0 or amount < 2 or N < 2 or N > 10000 or eps > 1:
    print('error input')
