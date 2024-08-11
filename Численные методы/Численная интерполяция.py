import numpy as np
from matplotlib import pyplot as plt

a = 0  # float(input())
b = 10  # float(input())
N = 59 # int(input())
xn = np.linspace(a, b, N - 1)
f = np.linspace(a, b, N)
fz = np.linspace(a, b, N - 1)  # f in xn
fn = np.linspace(a, b, N - 1)  # f -> L
ft = np.linspace(a, b, N - 1)  # f = Teilor
print('Интерполяция f(x)=cos((e^(x:2)):25) *5 вар* на отрезке (0, 10) используя полиом Лагранжа для разных сеток')


def L(xn, x, fz, N):
    p = np.ones(N)
    s = 0
    for i in range(N):  # sum{i=0->n}
        for j in range(N):  # prod{j=0->n, j!=i}
            if j != i:
                p[i] = p[i] * (xn - x[j]) / (x[i] - x[j])
        s += f[i] * p[i]
    return s


def o(fz, fn, N):
    S = 0
    Sn = 0
    for i in range(N-1):  # sum{i=1->n}
        S += (fz[i] - fn[i]) ** 2
    Sn = np.sqrt(S/(N-1))
    return Sn


x = np.linspace(a, b, N)


for i in range(N - 1):  # count xn = x-1
    xn[i] = (x[i+1] + x[i]) / 2
for i in range(N):  # count f = count x
    f[i] = np.cos(np.exp(x[i] / 2) / 25)
for i in range(N - 1):  # count fn = count xn
    fn[i] = L(xn[i], x, f, N)
for i in range(N-1):  # count fn = count f-1
    fz[i] = np.cos(np.exp(xn[i] / 2) / 25)
for i in range(N-1):  # count ft = count f
    ft[i] = 1 - (2.7 ** xn[i]) / 1250 + (2.7 ** (2 * xn[i])) / 9380000
print('1\n', 'Отклонение =', o(fz, fn, N))
q1 = o(fz, fn, N)


plt.plot(xn, fz, label="Истин")
plt.plot(xn, fn, label="Интерп")
plt.legend()
plt.xlabel(' x')
plt.ylabel(' f')
plt.title("Графики равномерной сетки")
plt.show()


for i in range(N):
    x[i] = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * (i + 1) - 1) / (2 * N) * np.pi)  # i+1 -> x[0]!=x[1]
for i in range( N - 1):
    xn[i] = (x[i+1] + x[i]) / 2
for i in range(N):
    f[i] = np.cos(np.exp(x[i] / 2) / 25)
for i in range(N - 1):
    fn[i] = L(xn[i], x, f, N)
for i in range(N - 1):
    fz[i] = np.cos(np.exp(xn[i] / 2) / 25)


print('2\n', 'Отклонение =', o(fz, fn, N))
q2 = o(fz, fn, N)


plt.plot(xn, fz, label="Истин")
plt.plot(xn, fn, label="Интерп")
plt.legend()
plt.xlabel(' x')
plt.ylabel(' f')
plt.title("Графики на узлах Чебышева")
plt.show()


print('Сравнение сеток:', f"{q1 - q2 : 3.3e}")
