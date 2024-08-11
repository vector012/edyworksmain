import numpy as np
from matplotlib import pyplot as plt
#5

def Q(X,T):
  return 2
def A(X,T):
  return 1
def F(X,T):
  return -2*np.cos(T-X)
def fi(X):
  return 2*np.cos(X)
def ksi(X):
  return 2*np.sin(X)
def alfa1(T):
  return -1
def betta1(T):
  return 1
def gamma1(T):
  return 2*(np.cos(T)-np.sin(T))
def alfa2(T):
  return 0
def betta2(T):
  return 1
def gamma2(T):
  return 2*np.cos(1-T)
def U0(X,T):
  return 2*np.cos(T-X)


def poshol1(N,K):
  U=np.zeros([N,K])
  for nn in range(N):
    if nn==0:
      U[nn,0]=fi(x[0])
      U[nn,K-1]=fi(x[K-1])
    if nn==1:
      U[nn,0]=fi(x[0])+ksi(x[0])*tay
      U[nn,K-1]=fi(x[K-1])+ksi(x[K-1])*tay
    for kk in range(1,K-1):
      if nn==0:
        U[nn,kk]=fi(x[kk])
      elif nn==1:
        U[nn,kk]=fi(x[kk])+ksi(x[kk])*tay
      else:
        U[nn,kk]=(tay**2*A(x[kk],t[nn-1]))/(Q(x[kk],t[nn-1])*h**2)*(U[nn-1,kk+1]-2*U[nn-1,kk]+U[nn-1,kk-1])+F(x[kk],t[nn-1])*tay**2/Q(x[kk],t[nn-1])+2*U[nn-1,kk]-U[nn-2,kk]
    if nn>0:
      U[nn,0]=(gamma1(t[nn])-alfa1(t[nn])*U[nn,1]/h)/(betta1(t[nn])-alfa1(t[nn])/h)
      U[nn,K-1]=gamma2(t[nn])
  return U
def poshol2(N,K):
  U=np.zeros([N,K])
  for nn in range(N):
    if nn==0:
      U[nn,0]=fi(x[0])
      U[nn,K-1]=fi(x[K-1])
    if nn==1:
      U[nn,0]=fi(x[0])+ksi(x[0])*tay+tay**2/2*(F(x[0],t[nn-1])/Q(x[0],t[nn-1])-A(x[0],t[nn-1])/Q(x[0],t[nn-1])*(fi(x[0])-2*fi(x[1])+fi(x[2]))/h**2)
      U[nn,K-1]=fi(x[K-1])+ksi(x[K-1])*tay+tay**2/2*(F(x[K-1],t[nn-1])/Q(x[K-1],t[nn-1])-A(x[K-1],t[nn-1])/Q(x[K-1],t[nn-1])*(fi(x[K-3])-2*fi(x[K-2])+fi(x[K-1]))/h**2)
    for kk in range(1,K-1):
      if nn==0:
        U[nn,kk]=fi(x[kk])
      elif nn==1:
        U[nn,kk]=fi(x[kk])+ksi(x[kk])*tay+tay**2/2*(F(x[kk],t[nn-1])/Q(x[kk],t[nn-1])-A(x[kk],t[nn-1])/Q(x[kk],t[nn-1])*(fi(x[kk-1])-2*fi(x[kk])+fi(x[kk+1]))/h**2)
      else:
        U[nn,kk]=(tay**2*A(x[kk],t[nn-1]))/(Q(x[kk],t[nn-1])*h**2)*(U[nn-1,kk+1]-2*U[nn-1,kk]+U[nn-1,kk-1])+F(x[kk],t[nn-1])*tay**2/Q(x[kk],t[nn-1])+2*U[nn-1,kk]-U[nn-2,kk]
    if nn>0:
      U[nn,0]=(gamma1(t[nn])-alfa1(t[nn])*U[nn,2]/h/2)/(betta1(t[nn])-alfa1(t[nn])/h)
      U[nn,K-1]=gamma2(t[nn])
  return U
a=0
b=1
vvodim=2
if vvodim == 1:
  tay=float(input("                           = "))
  T=float(input("                        = "))
else:
  tay=0.05
  T=1
h=0.05
n=int(T/tay+1)
k=int((b-a)/h+1)
x=np.zeros([k])
for i in range(k):
  x[i]=a+i*h
t=np.zeros([n])
for i in range(n):
  t[i]=i*tay
Uorig=np.zeros([n,k])
U1=np.zeros([n,k])
U2=np.zeros([n,k])
U3=np.zeros([n,k])
U1=poshol1(n,k)
U2=poshol2(n,k)



Y=np.zeros([n,k])
X=np.zeros([n,k])
for nn in range(n):
  for kk in range(k):
    Y[nn,kk]=t[nn]
    X[nn,kk]=x[kk]
    Uorig[nn,kk]=U0(x[kk],t[nn])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Uorig)
ax.plot_surface(X, Y, U1)
ax.plot_surface(X, Y, U2)

plt.show()


otkl1=np.zeros([n,k])
otkl2=np.zeros([n,k])
for nn in range(n):
  for kk in range(k):

    otkl1[nn,kk]=abs(U0(x[kk],t[nn])-U1[nn,kk])
    otkl2[nn,kk]=abs(U0(x[kk],t[nn])-U2[nn,kk])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, otkl1)
ax.plot_surface(X, Y, otkl2)
plt.show()
ath=A(0,0)/Q(0,0)*tay**2/h**2
if ath<1:
  print("a^2 = ",A(0,0)/Q(0,0),", tay = ",tay,", h = ",h)
  print("a*tay/h = ",ath,"< 1")
  print("max otkl1 = ", otkl1[n-1,k-1])
  print("max otkl2 = ", otkl2[n-1,k-1])
else:
  print("a^2 = ",A(0,0)/Q(0,0),", tay = ",tay,", h = ",h)
  print("a*tay/h = ",ath,"> 1")