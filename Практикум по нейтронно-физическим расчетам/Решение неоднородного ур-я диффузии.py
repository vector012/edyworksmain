import numpy as np
from matplotlib import pyplot as plt

# ур диф с источником в 1Д плоская геометрия слева абс. отражатель справа вакуум
n = 10**7
Ea  = np.zeros(n)
Es = np.zeros(n)
Et = np.zeros(n)
D = np.zeros(n)
Ea[0] = 0.03
Es[0] = 0.3
A = 1
C = 0
for i in range(1, n):
  Ea[i] = Ea[i-1] + C
for i in range(1, n):
  Es[i] = Es[i-1] + C
Et = Es + Ea
D = 1 / (3*Es*(1-2/(A*3)))
H = 165
gr = 100**100
x = np.linspace(0,H,n)


F = np.zeros(n)
h = (x[1] - x[0])
a = np.zeros(n)
for i in range(n):
  a[i] = - (D[i-1] + D[i]) / (2*h)
c = a
gl = 0
b1 = Ea[0] * h - c[0] + gl/2 #отражение
b = Ea * h - c - a
bN = Ea[-1] * h - a[-1] + gr/2
A = np.zeros(n)
q = 1
A[0] = (q * h) / b1
B = np.zeros(n)
B[0] = a[0] / b1
F = np.zeros(n)
F[n - 1] =  A[n - 1]
for i in range(1, n ):
  A[i] = (h - a[i] * A[i - 1]) / (b[i] - a[i] * B[i - 1])
  B[i] = a[i] / (b[i] - a[i] * B[i -1])
A[n-1] = (h - a[n-1]) / bN
for i in range(n - 2, -1, -1):
  F[i] = A[i] - B[i] * F[i + 1]
plt.plot(x, F/max(F), label="progon")
F_analit = np.zeros(n)
L = np.sqrt(D / Ea)
for i in range(n):
  F_analit[i] = 1 / Ea[i] * (1 - (np.cosh(x[i] / L[i]) / np.cosh(H / L[i])))


plt.plot(x, F_analit/max(F_analit), label="analit")
plt.xlabel('x')
plt.ylabel('Ф')
plt.title('График нормированных потоков')
plt.legend()
plt.show()


si = np.zeros(n-1)
for i in range(n-1):
  si[i] = np.abs(F[i]-F_analit[i])/F_analit[i]*100
plt.plot(x[:-1:], si, label = 'otclon')
plt.xlabel('X')
plt.ylabel('error')
plt.show()
