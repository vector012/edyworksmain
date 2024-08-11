import numpy as np
import matplotlib.pyplot as plt


#1
def f(x):
  return (np.tan(np.tanh(x))-np.sinh(np.cos(0.5*x))-1)


eps = 10**(-3)
xr = 10
xl = 0
xm = 5
c = 0
f_r = f(xr)
f_l = f(xl)
f_m = f(xm)


while np.abs(f_m) > eps:
  c += 1
  if f_l*f_m < 0:
    xr = xm
    f_r = f(xr)
    xm = (xl+xm)/2
    f_m = f(xm)
  elif f_m*f_r < 0:
    xl = xm
    xm = (xr+xm)/2
    f_l=f(xl)
    f_m=f(xm)
r = xm
L = c


print("1 bisection: \n", "    eps = ", eps, "    x* = ", f'{r:.13}00000000', "    f(x*) = ", f'{f(r):.4}', "     iters  = ", c)


eps = 10**(-6)
xr = 10
xl = 0
xm = 5
c = 0
f_r = f(xr)
f_l = f(xl)
f_m = f(xm)


while (xr-xl)/2 > eps:
  c += 1
  if f_l*f_m < 0:
    xr = xm
    f_r = f(xr)
    xm = (xl+xm)/2
    f_m = f(xm)
  elif f_m*f_r < 0:
    xl = xm
    xm = (xr+xm)/2
    f_l=f(xl)
    f_m=f(xm)
r = xm


print("     eps = ", eps, "    x* = ",  f'{r:.13}', "    f(x*) = ", f'{f(r):.4}', "    iters  = ", c)

eps = 10**(-9)
xr = 10
xl = 0
xm = 5
c = 0
f_r = f(xr)
f_l = f(xl)
f_m = f(xm)


while np.abs(f_m) > eps:
  c += 1
  if f_l*f_m < 0:
    xr = xm
    f_r = f(xr)
    xm = (xl+xm)/2
    f_m = f(xm)
  elif f_m*f_r < 0:
    xl = xm
    xm = (xr+xm)/2
    f_l=f(xl)
    f_m=f(xm)
r = xm


print("     eps = ", eps, "    x* = ",  f'{r:.13}', "    f(x*) = ", f'{f(r):.4}', "    iters  = ", c)


eps = 10**(-12)
xr = 10
xl = 0
xm = 5
c = 0
f_r = f(xr)
f_l = f(xl)
f_m = f(xm)


while np.abs(f_m) > eps:
  c += 1
  if f_l*f_m < 0:
    xr = xm
    f_r = f(xr)
    xm = (xl+xm)/2
    f_m = f(xm)
  elif f_m*f_r < 0:
    xl = xm
    xm = (xr+xm)/2
    f_l=f(xl)
    f_m=f(xm)
r = xm


print("     eps = ", eps, "    x* = ",  f'{r:.13}', "    f(x*) = ", f'{f(r):.4}', "    iters  = ", c)

#2
def df(x):
  return (((np.cosh(np.cos(x/2)))*np.sin(x/2)*((np.cosh(x))**2)*((np.cos(np.tanh(x)))**2)+2)/(2*((np.cosh(x))**2)*(np.cos(np.tanh(x)))**2))


eps = 10**(-3)
xm = 5
cn = 0
x0 = xm


while np.abs(f(x0)) > eps:
  x0 = x0 - (f(x0)/df(x0))
  cn += 1
rn = x0
M = cn


print("2 newton: \n", "    eps = ", eps, "    x* = ", f'{rn:.13}', "    f(x*) = ", f'{f(rn):.4}', "     iters  = ", cn)


eps = 10**(-6)
xm = 5
cn = 0
x0 = xm


while np.abs(f(x0)) > eps:
  x0 = x0 - (f(x0)/df(x0))
  cn += 1
rn = x0


print("     eps = ", eps, "    x* = ", f'{rn:.13}', "    f(x*) = ", f'{f(rn):.4}', "     iters  = ", cn)


eps = 10**(-9)
xm = 5
cn = 0
x0 = xm


while np.abs(f(x0)) > eps:
  x0 = x0 - (f(x0)/df(x0))
  cn += 1
rn = x0


print("     eps = ", eps, "    x* = ", f'{rn:.13}', "    f(x*) = ", f'{f(rn):.4}', "     iters  = ", cn)


eps = 10**(-12)
xm = 5
cn = 0
x0 = xm


while np.abs(f(x0)) > eps:
  x0 = x0 - (f(x0)/df(x0))
  cn += 1
rn = x0


print("     eps = ", eps, "    x* = ", f'{rn:.13}', "    f(x*) = ", f'{f(rn):.4}', "      iters  = ", cn)

#3
def F():
  x = np.linspace(0, 10, 10**3)
  return f(x)


F = F()


plt.plot(np.linspace(0, 10, 10**3), F, color = 'black')
plt.plot(np.linspace(0, 10, 10**3), F*0, color = 'red')
plt.grid()
plt.title('graf of func')
plt.xlabel('x')
plt.ylabel('f')


print("For eps = 0.001: itB - itN = ", L - M)
