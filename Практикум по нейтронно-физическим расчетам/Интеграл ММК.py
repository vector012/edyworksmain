import numpy as np
import matplotlib.pyplot as plt
import numpy.random as ran
from sympy import *

X_y = []  # yellow
Y_y = []
X_b = []  # black
Y_b = []
M, N, A, B = 500, 1000, -3, 8  # int(input())                ### ИСПРАВИТЬ ВРУЧНУЮ!

def MonteCarlo(n, a, b):  # фигачем точки на плоскости
  count, H = 0, 0
  X, Y = np.zeros(n), np.zeros(n)
  for i in range(n):
    x = ran.uniform(a, b)   # x in [a, b]
    c = -100                                                ### ИСПРАВИТЬ ВРУЧНУЮ!
    f = 4 * x ** 3 + c                                      ### ИСПРАВИТЬ ВРУЧНУЮ!
    '''Если график начинается не из 0, а выше, то площадь под ним будет не вся,
          поэтому вводим проверку-поправку: только при a > 0 перед константой
     ПРОТИВОПОЛОЖНЫЙ ЗНАК тому, что в формуле функции, т.е. y=x+c -> x-c<0'''
    if                 4 * a ** 3 + c <= 0:                 ### ИСПРАВИТЬ ВРУЧНУЮ!
      y = ran.uniform((4 * a ** 3 + c), (4 * b ** 3 + c))   ### ИСПРАВИТЬ ВРУЧНУЮ!
    else:
      y =                ran.uniform(0, (4 * b ** 3 + c))   ### ИСПРАВИТЬ ВРУЧНУЮ!
      H += 1
    X[i], Y[i] = x, y
    if 0 <= y <= f: # good
      count += 1
      X_y.append(x)
      Y_y.append(y)
    elif 0 < y > f:
      X_b.append(x)
      Y_b.append(y)
    elif 0 > y < f:
      X_b.append(x)
      Y_b.append(y)
    else: # good but with '-' becouse y<0
      count -= 1
      X_y.append(x)
      Y_y.append(y)
  S = (max(X) - min(X)) * (max(Y) - min(Y))  # размер прямоугольника
  ny = count / n  # доля нужных точек
  integ = S * ny  # результат
  #print(H)
  return integ

h = 0
I = np.zeros(M)
while h < M:  # число испытаний
  I[h] = MonteCarlo(N, A, B)
  h += 1
result = str(np.round(np.average(I), 1))
error = str(np.round(0.674 * np.nanstd(I), 1))
dots = str(N * M)
f_x = np.linspace(A - 0.1, B + 0.1, N)  # строю графики
init_printing(use_unicode=False, wrap_line=False)
x = Symbol('x')
SS = integrate(4 * x ** 3 - 100, x)                       ### ИСПРАВИТЬ ВРУЧНУЮ!
R =  integrate(4 * x ** 3 - 100, (x, A, B))               ### ИСПРАВИТЬ ВРУЧНУЮ!
F =str('f(x) = 4 * x ** 3 - 100')                         ### ИСПРАВИТЬ ВРУЧНУЮ!
f_y     =    4 * f_x ** 3 - 100                           ### ИСПРАВИТЬ ВРУЧНУЮ!
print('Численный рассчёт: F(x) =', SS, '=', np.round(float(R), 1), 'при x на [', A, ';', B, ']')

A_F, B_F = str(A), str(B)
if min(f_y) > 0:
  m1 = 0
else:
  m1 = min(f_y)
if max(f_y) < 0:
  m2 = 0
else:
  m2 = max(f_y)
y_inf = np.linspace(m1, m2, N)
if min(f_x) > 0:
  m3 = 0
else:
  m3 = min(f_x)
if max(f_x) < 0:
  m4 = 0
else:
  m4 = max(f_x)
x_inf = np.linspace(m3, m4, N)
plt.plot(x_inf, 0 * f_x, linewidth = 2, linestyle = '-', color = 'blue', label = 'Оси')
plt.plot(0 * y_inf, y_inf, linewidth = 2, color = 'blue')
plt.plot(f_x, f_y, linewidth = 2, color = 'red', label = 'Фун-ия')
plt.scatter(X_y, Y_y, s = 1, c = 'yellow', label = 'Попало')
plt.scatter(X_b, Y_b, s = 1, c = 'black', label = 'Мимо')
plt.title('Интеграл через Монте Карло: ' + result + '+-' + error + ' при a = 0.5')
plt.suptitle('Всего точек = ' + dots + '. ' + F + ' при x на [' + A_F + ';' + B_F + ']')
plt.xlabel(' X')
plt.ylabel(' Y')
plt.text(min(f_x), max(f_y) + 1, '>Изображены оси с графиком и закрашенная площадь под ним<', color = 'm')
plt.legend()
plt.show()
print("К сожалению на краях графика иногда могут вылезать ложные точки, а также",
"лишние точки вдоль графика самой функции, но это лишь визуальная ошибка, не обрайте внимания!")
