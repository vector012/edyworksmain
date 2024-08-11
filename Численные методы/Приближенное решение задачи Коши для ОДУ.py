import numpy as np
from matplotlib import pyplot as plt


a, b = 0, 1   #gran ysl
N = 20
h = 1/N


def U0(x):    #analit vid
 return (np.log(1+x) + np.tanh(x))


def f(u, x, v):
 return ((1/(np.cosh(x))**2)*(1/(1+x)+2*np.log(1+x))-1*v/(1+x)-2*u/((np.cosh(x))**2))


def V0(x):   #proizvodnaia
  return (1/(1+x) + (1/(np.cosh(x)))**2)


def Z(v):
  return v


def Ob_Euler(N):
  h = 1/N
  U = np.zeros(N+1)
  x = a
  u = U0(0)
  v = V0(0)
  U[0] = u
  for i in range(N):
    u += h*Z(v)
    v += h*f(u, x, v)
    x += h
    U[i+1] = u
  return(U)


def Clas_Runge_kutta(N):
  h = 1/N
  U = np.zeros(N+1)
  V = np.zeros(N+1)
  x = a
  u = U0(0)
  v = V0(0)
  U[0] = u
  V[0] = v
  for i in range(N):
    k1 = Z(v)
    q1 = f(u, x, v)
    k2 = Z(v + q1*h/2)
    q2 = f(u + k1*h/2, x + h/2, v + q1*h/2)
    k3 = Z(v + q2*h/2)
    q3 = f(u + k2*h/2, x + h/2, v + q2*h/2)
    k4 = Z(v + q3*h)
    q4 = f(u + k3*h, x + h, v + q3*h)
    u += h/6*(k1 + 2*k2 + 2*k3 + k4)
    v += h/6*(q1 + 2*q2 + 2*q3 + q4)
    x += h
    U[i+1] = u
    V[i+1] = v
  return U, V


def Yavn_Adams(N):
  h = 1/N
  U = np.zeros(N+1)
  x = np.array([0,h,2*h])
  u, v = Clas_Runge_kutta(N)
  U[0] = u[0]
  U[1] = u[1]
  U[2] = u[2]
  for i in range(N-2):
    q1 = f(u[2],x[2],v[2]) * h
    q2 = f(u[1],x[1],v[1]) * h
    q3 = f(u[0],x[0],v[0]) * h
    k1 = Z(v[2]) * h
    k2 = Z(v[1]) * h
    k3 = Z(v[0]) * h
    u[0] = u[1]
    u[1] = u[2]
    u[2] = u[2] + (23*k1-16*k2+5*k3)/12
    U[i+3] = u[2]
    v[0] = v[1]
    v[1] = v[2]
    v[2] = v[2] + (23*q1-16*q2+5*q3)/12
    x += h
  return(U)


x = np.arange(0,1+h,h)
u_an = U0(x)
u, v = Clas_Runge_kutta(N)


plt.plot(x, Ob_Euler(N), label="euler")
plt.plot(x, u, label="runge")
plt.plot(x, Yavn_Adams(N), label="adams")
plt.plot(x, u_an, label="analit")
plt.legend()
plt.title("Графики численных решений на x = (0;1)")
plt.show()


n = 500 #число шагов
M = np.zeros(n)
h = np.zeros(n)
M[0] = 100
for i in range(n-1):
  M[i+1] = M[i] + 1
h = 1/(M)
otkl_euler = np.zeros(n)
otkl_runge = np.zeros(n)
otkl_adams = np.zeros(n)
for j in range(n):
  u_an = U0(1.0)
  u_n = Ob_Euler(int(M[j]))
  otkl_euler[j] = np.abs(u_n[-1] - u_an)
  u_n, v = Clas_Runge_kutta(int(M[j]))
  otkl_runge[j] = np.abs(u_n[-1] - u_an)
  u_n = Yavn_Adams(int(M[j]))
  otkl_adams[j] = np.abs(u_n[-1] - u_an)


plt.plot(np.log(h), np.log(otkl_euler), label="euler")
plt.plot(np.log(h), np.log(otkl_runge), label="runge")
plt.plot(np.log(h), np.log(otkl_adams), label="adams")
plt.legend()
plt.title("Графики отклонений в логарифмическом масштабе при x=1")
plt.show()


print('euler O(x)', (np.log(otkl_euler[0])-np.log(otkl_euler[-1]))/((np.log(h[0])-np.log(h[-1]))))
print('runge O(x)', (np.log(otkl_runge[0])-np.log(otkl_runge[-1]))/((np.log(h[0])-np.log(h[-1]))))
print('adams O(x)', (np.log(otkl_adams[0])-np.log(otkl_adams[-1]))/((np.log(h[0])-np.log(h[-1]))))


u1_euler = Ob_Euler(N)
u1_runge, v = Clas_Runge_kutta(N)
u1_adams = Yavn_Adams(N)
u2_euler = Ob_Euler(2*N)
u2_runge, v = Clas_Runge_kutta(2*N)
u2_adams = Yavn_Adams(2*N)
print('\nRunge rule')
print(np.abs(u2_euler[-1]  - u1_euler[-1]),'euler')
print(np.abs(u2_runge[-1]  - u1_runge[-1])/15,'runge')
print(np.abs(u2_adams[-1]  - u1_adams[-1])/7,'adams')


print()
u2_real = U0(1)
print(np.abs(u2_euler[-1] - u2_real), 'eul')
print(np.abs(u2_adams[-1] - u2_real), 'adam')
print(np.abs(u2_runge[-1] - u2_real), 'rung')

