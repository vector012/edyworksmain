import numpy as np
from matplotlib import pyplot as plt

print('Guba F(x)=1/(x^3+x+10) x=[-1,1] ')
'''
#0
def inpu(N):
  def X():
    x = np.zeros(N)
    x[0] = a
    x[N - 1] = b
    for i in range(1, N - 1):
      x[i] += (b - a) / N
    return X()
  def F():
    f = np.zeros(N)
    for i in range(N):
      f[i] += 1 / (x[i] ** 3 + x[i] + 10)
    return F()

def error(I, RI):
  erro = np.abs(I - RI)
  return erro
'''

#1
def int_rect_l(F, h):  #left with step
  integ = 0
  for i in range(len(F) - 1):
    integ += F[i] * h
  return integ


def int_rect_r(F, x):  # right with x
  integ = 0
  for i in range(1, len(F)):
    integ += F[i] * (x[i] - x[i - 1])
  return integ


#2
def int_trap(F, h):  # Basic
  integ = 0
  for i in range(len(F) - 1):
    integ += (F[i + 1] + F[i]) * h / 2
  return integ


def int_trap_K(F, h):  # Kotesa
  integ = h * (F[0] + F[N - 1]) / 2
  for i in range(1, len(F) - 1):
    integ += F[i] * h
  return integ


#3
def int_simp(F, h):  # Kotesa
  integ, i = 0, 0
  for j in range(len(F) // 2 ):
    integ = integ + h / 3 * ( F[i] + 4 * F[i + 1] + F[i + 2])
    i += 2
  return integ


def int_simp_2(F):  # Kotesa with X
  integ = 0
  for i in range(1, int(len(F) / 2)): #F[0] doesn't included (((
    integ += 2 * F[2 * i] + 4 * F[2 * i - 1]
  return (x[2] - x[0]) * (integ + F[0] + F[N - 1]) / 6


def int_simp_3(F, integr = 0):  # Kotesa another form
  for k in range(1, len(F), 2):
    integr += F[k-1] + 4*F[k] + F[k+1]
  return (x[2] / 3 - x[1] / 3) * integr


def int_simp_4(F, h, N):  # interpol WTF
  #integ = (b - a) / 6 * (F[0] + 4*F[N/2] + F[N - 1])
  integ = N * h * (F[0] + 4 * F[ N // 2 - 1 ] + F[N - 1]) / 6
  return integ


n = input('N = ')
h = np.zeros(int(n))
M = np.zeros(int(n))
M[0] = 3
for i in range(int(n) - 1):
  M[i + 1] = M[i] + 2
err1 = np.zeros(int(n))
err2 = np.zeros(int(n))
err3 = np.zeros(int(n))
err4 = np.zeros(int(n))
err5 = np.zeros(int(n))
err6 = np.zeros(int(n))
a = -1
b = -a
accurately = (3 * np.pi + np.log(104976)) / 104  # Wolfram chisl
int_F_a = (-np.log(a ** 2 - 2 * a + 5) + 2 * np.log(a + 2) + 3 * np.arctan((a - 1) / 2)) / 26
int_F_b = (-np.log(b ** 2 - 2 * b + 5) + 2 * np.log(b + 2) + 3 * np.arctan((b - 1) / 2)) / 26
In_F = int_F_b - int_F_a
for j in range(int(n)):
  N = int(M[j])
  x = np.linspace(a, b, N)
  h[j] = 2 * b / (M[j] - 1)
  F = 1 / (x ** 3 + x + 10)

  F1 = int_rect_l(F, h[j])
  err1[j] = abs(F1 - accurately)
  F5 = int_rect_r(F, x)
  err5[j] = abs(F5 - accurately)

  F2 = int_trap_K(F, h[j])
  err2[j] = abs(F2 - In_F)
  F6 = int_trap(F, h[j])
  err6[j] = abs(F6 - accurately)

  F3 = int_simp(F, h[j])
  err3[j] = abs(F3 - accurately)
  F4 = int_simp_3(F)
  err4[j] = abs(F4 - In_F)

plt.plot(np.log(h), np.log(err1),  label="rect")
plt.plot(np.log(h), np.log(err2),  label="trap")
plt.plot(np.log(h), np.log(err3), label="simp")
plt.xlabel('h')
plt.ylabel('error')
#plt.plot(np.log(h), np.log(err5),  label="rect2", color = 'red')
#plt.plot(np.log(h), np.log(err6),  label="trap2", color = 'yellow')
#plt.plot(np.log(h), np.log(err4),  label="simp2", color = 'black')
err10 = np.zeros(int(n))
F10 = np.zeros(int(n))
for j in range(int(n)):
  F10 = int_simp_4(F, h, int(n))
  err10[j] = np.abs(F10[j] - (accurately + In_F) / 2)
plt.plot(np.log(h), np.log(err10),  label="analytic", color = 'gray')
plt.legend()
plt.show()
#k
K = np.zeros(int(n)-20)
print('error for sims =', err3[int(n)-1])
for i in range(10, int(n)-10):
  K[i-10] = (np.log(err3[i-10]) - np.log(err3[i])) / (np.log(h[i-10]) - np.log(h[i]))
print('k for simps = ', np.mean(K), '+/-', np.std(K))
