# Импорт требуемых библиотек
import numpy as np
from matplotlib import pyplot as plt

# Условие:
print('Guba F(x)=cosh(e^-x^2) x=[-5,5]\n')
A = -5  # float(input()) # Левая раницf отрезка
B = 5  # float(input())  # Правая границы отрезка
M = 800  # int(input()) число шагов

# 1 Пункт

# Вычисление аналитической производной
'''
from sympy import diff, symbols, cosh, exp
y = symbols('y')
d_f_a = diff(cosh(exp(-(y ** 2))), y)
print('\nAnalytic diff F(y) =', d_f_a,
      '\n' if d_f_a != 0 else 'The function is not differentiable')
'''

# Создаю функции для рассчёта
# Правая функций для расчёта первой производной(ниже точность)
def diff_1_right(func, step, quan):
    dif_func = np.zeros(quan - 1)  # Для крайней правой точки вычислять производную не нужно
    for i in range(quan - 1):
        dif_func[i] = (func[i + 1] - func[i]) / step  # Просто по формуле
    return dif_func
# Центральная
def diff_1_central(func, step, quan):
    dif_func = np.zeros(quan - 2)  # Для крайних точек вычислять производную не нужно
    for i in range(quan - 2):
        dif_func[i] += (func[(i + 1) + 1] - func[(i + 1) - 1]) / (2 * step)  # Просто по формуле
    return dif_func
# Заполнение массива функции
def in_put(func, value, quan):
    for i in range(quan):
        func[i] = np.cosh(np.exp(-(value[i] ** 2)))  # Просто по формуле
    return func
# Среднеквадратичное отклонение
def standard_deviation(func_chis, func_anal, quan):
    amount = 0
    for i in range(quan - 1):
        amount += (func_chis[i] - func_anal[i]) ** 2  # Просто по формуле
    deviation = np.sqrt(amount / (quan - 1))
    return deviation
# Заполнение массива функции
def analytic_func_in(func_anal, value, quan):  # Заполняется массив значений функции аналитической массивом равномерых иксов в зависимости от числа узлов
    for i in range(quan):
        func_anal[i] = -2 * np.exp(-(value[i] ** 2)) * value[i] * np.sinh(np.exp(-(value[i] ** 2)))  # Просто по формуле
    return func_anal
# Просто задаю точки и выполняю рассчёт
h = np.zeros(M)  # массив шагов (М штук)
L = np.zeros(M)  # массив чисел узлов (М штук int)
L[0] = 5
std_r = np.zeros(M)  # массив значинеий ско для правой производной отвечающих итому элементу шага (М штук)
std_c = np.zeros(M)  # массив значинеий ско для центральной производной отвечающих итому элементу шага (М штук)
for i in range(M - 1):
    L[i + 1] = L[i] + 1  # Наращиваю число узлов
# Задаю шаг
if A != B and M != 0:
    for j in range(M):
        h[j] = (B - A) / (int(L[j]) - 1)
else:
    print('err inp')
# Пробигая все М шагов выполняем задание:
for k in range(M):
    N = int(L[k])
    # Задание массивов под узлы:
    x = np.linspace(A, B, N) # Задание равномерной сетки
    f_r = np.zeros(N)
    f_c = np.zeros(N)
    a_f = np.zeros(N)
    # Набиваю функции значениями иксов
    in_put(f_r, x, N)
    in_put(f_c, x, N)
    analytic_func_in(a_f, x, N)
    # Дифференцирую функции
    f_r = diff_1_right(f_r, h[k], N)
    f_c = diff_1_central(f_c, h[k], N)
    a_f = np.delete(a_f, -1)
    a_f = np.delete(a_f, 0)
# Вывод массива данных производных  print(f_c)
    # Рассчёт СКО
    std_r[k] = standard_deviation(f_r, a_f, N - 1)
    std_c[k] = standard_deviation(f_c, a_f, N - 2)
# Построение графиков
plt.plot(np.log(h), np.log(std_r), label="right")  # Строю график 1
plt.plot(np.log(h), np.log(std_c), label="central")  # Строю график 2
plt.legend()  # Подписи по красоте
plt.xlabel('h')
plt.ylabel('std')
plt.title("Графики первых производных")   # Название
plt.show()  # Чтобы вывело графики

#k=
s1 = np. zeros(N-200)
s2 = np. zeros(N-200)
for i in range(100, N-100):
  s1[i-100] = (np.log(std_r[i-10]) - np.log(std_r[i])) / (np.log(h[i-10]) - np.log(h[i]))
  s2[i-100] = (np.log(std_c[i-10]) - np.log(std_c[i])) / (np.log(h[i-10]) - np.log(h[i]))
print(np.mean(s1), '+-', 2*np.std(s1))
print(np.mean(s2), '+-', 2*np.std(s2))
# 2 Пункт

# Создаю функции для рассчёта
# Вторая производная второго порядка точности
def diff_2_ord2(func, step, quan):
    dif_func = np.zeros(quan - 2)
    for i in range(quan - 2):
        dif_func[i] = (func[(i + 1) + 1] - 2 * func[(i + 1)] + func[(i + 1) - 1]) / (step * step) # Просто по формуле
    return dif_func
# Вторая производная 4 порядка точности
def diff_2_ord4(func, step, quan):
    dif_func = np.zeros(quan - 4)
    for i in range(quan - 4):
        dif_func[i]=(-func[i+2+2]+16*func[i+1+2]-30*func[2+i]+16*func[i+2-1]-func[i-2+2])/(12*step**2)
    return dif_func
# Аналитический расчёт 2 произв
def anal_input(fun_anal, val):
    for i in range(len(fun_anal)-1):
        fun_anal[i]=4*np.exp(-(val[i]**2))*(val[i]**2)*np.sinh(np.exp(-(val[i]**2)))-2*np.exp(-(val[i]**2))*np.sinh(np.exp(-(val[i]**2)))+4*np.exp(-2*(val[i]**2))*(val[i]**2)*np.cosh(np.exp(-(val[i]**2)))
    return fun_anal
# Среднеквадратичное отклонение
def standard_deviation(func_chis, func_anal, quan):
    amount = 0
    for i in range(quan):
        amount = amount + (func_chis[i] - func_anal[i]) ** 2  # Просто по формуле
    deviation = np.sqrt(amount / (quan - 1))
    return deviation
# Заполнение функций
def inp(func, value):
    for i in range(len(func)):
        func[i] = np.cosh(np.exp(-(value[i] ** 2)))
    return func  # Возвращаем итог
# Просто задаю точки и выполняю рассчёт
h = np.zeros(M)  # массив шагов (М штук)
L = np.zeros(M)  # массив чисел узлов (М штук int)
L[0] = 10  # начинается с 10 точек на оси икс
std_2 = np.zeros(M)  # массив значинеий ско для правой производной отвечающих итому элементу шага (М штук)
std_4 = np.zeros(M)  # массив значинеий ско для центральной производной отвечающих итому элементу шага (М штук)
for i in range(M - 1):
    L[i + 1] = L[i] + 1  # Наращиваю число узлов
# Задаю шаг
if A != B and M != 0:
    for j in range(M):
        h[j] = (B - A) / (int(L[j]) - 1)
else:
    print('err inp')
# Пробигая все М шагов выполняем задание:
for k in range(M):
    N = int(L[k])
    # Задание массивов под узлы:
    x = np.linspace(A, B, N)  # Задание равномерной сетки
    f_2 = np.zeros(N)
    f_4 = np.zeros(N)
    a_f = np.zeros(N)
    # Набиваю функции значениями иксов
    f_2 = inp(f_2, x)
    f_4 = inp(f_4, x)
    a_f = anal_input(a_f, x)
    # Дифференцирую функции
    f_2 = diff_2_ord2(f_2, h[k], N)
    f_4 = diff_2_ord4(f_4, h[k], N)
    a_f = a_f[1:-1]
    # Рассчёт СКО
    std_2[k] = standard_deviation(f_2, a_f, N - 4)
    a_f = a_f[1:-1]
    std_4[k] = standard_deviation(f_4, a_f, N - 4)

def standard_deviation(func_chis, func_anal, quan):
    amount = 0
    for i in range(quan):
        amount = amount + (func_chis[i] - func_anal[i]) ** 2  # Просто по формуле
    deviation = np.sqrt(amount / (quan - 1))
    return deviation
# Построение графиков
plt.plot(np.log(h), np.log(std_2), label="O(2)")  # Строю график 1
plt.plot(np.log(h), np.log(std_4), label="O(4)")  # Строю график 2
plt.legend()
plt.xlabel('h')
plt.ylabel('std')
plt.title("Графики вторых производных")   # Название
plt.show()  # Чтобы вывело графики

#k =
s3 = np. zeros(N-180)
s4 = np. zeros(N-180)
for i in range(90, N-90):
  s3[i-90] = (np.log(std_2[i-10]) - np.log(std_2[i])) / (np.log(h[i-10]) - np.log(h[i]))
  s4[i-90] = (np.log(std_4[i-10]) - np.log(std_4[i])) / (np.log(h[i-10]) - np.log(h[i]))
print(np.mean(s3), '+-', 2*np.std(s3))
print(np.mean(s4), '+-', 2*np.std(s4))

'''
#Шаг меняется от изменения границ
def X(left, right, quan):
    x = np.linspace(left, right, quan)
    return x
N = 25
a = np.zeros(M)
a[0] = A
for i in range(M-1):
    a[1+i] = a[i] - 1/N
b = np.zeros(M)
b[0] = B
for i in range(M-1):
    b[i+1] = b[i] + 1/N
h = np.zeros(M)
for i in range(M):
    h[i] = (b[i] - a[i]) / N
f_r = np.zeros(M)
f_c = np.zeros(M)
a_f = np.zeros(M)
std_r = np.zeros(M)
std_c = np.zeros(M)
for i in range(M):

    std_c[i] = standard_deviation(diff_1_central(in_put(f_c, X(a[i], b[i], N), N), h[i], N), analytic_func_in(a_f, X(a[i], b[i], N), N - 1), N - 1)
    std_r[i] = standard_deviation(diff_1_right(in_put(f_r, X(a[i],b[i], N), N), h[i], N), analytic_func_in(a_f, X(a[i], b[i], N), N), N)
plt.plot(np.log2(h), np.log2(std_r), label="r")
plt.plot(np.log2(h), np.log2(std_c), label="c")
plt.legend()
'''
