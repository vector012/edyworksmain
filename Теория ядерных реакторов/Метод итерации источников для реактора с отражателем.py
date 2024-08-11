import numpy as np
from matplotlib import pyplot as plt


# Условие Губа
print('Рассчёт Ф численным методом итерации источника в сферическом реакторе')
# const
print("\nТаблица 0:")
l = 8
print("l = ", l)
r_1 = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0]
print("r: ", r_1)
d_r_1 = np.zeros(l - 1)
for j in range(0, l - 1):
  d_r_1[j]=(r_1[j+1] - r_1[j])
print("delta r = ", d_r_1[0])
D_az_1 = [1.1, 1.1, 1.1, 1.1]
print("D az 1 = ", D_az_1[0])
R_az = 16
print("R az = ", R_az)
D_otr_1 = [0.69, 0.69, 0.69, 0.69]
print("D otr 1 = ", D_otr_1[0])
lamda_1 = [1.333, 1.037, 1.013, 1.007, 1.0, 1.0, 1.0, 1.0]
print("lambda 1 = ", lamda_1)
S_ad_az_1 = [0.085, 0.085, 0.085, 0.085]
print("Sigma ad az 1 = ", S_ad_az_1[0])
S_ad_otr_1 = [0.0135, 0.0135, 0.0135, 0.0135]
print("Sigma ad otr 1 = ", S_ad_otr_1[0])
Q_az_1 = [1.0, 1.0, 1.0, 1.0]
print("Q az 1 = ", Q_az_1[0])
Q_otr_1 = [0, 0, 0, 0]
print("Q otr 1 = ", Q_otr_1[0])
r_2 = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0]
d_r_2 = np.zeros(l - 1)
for j in range(0, l - 1):
  d_r_2[j]=(r_2[j+1] - r_2[j])
D_az_2 = [0.145, 0.145, 0.145, 0.145]
print("D az 2 = ", D_az_2[0])
D_otr_2 = [0.416, 0.416, 0.416, 0.416]
print("D otr 2 = ", D_otr_2[0])
lamda_2 = [1.333, 1.037, 1.013, 1.007, 1.0, 1.0, 1.0, 1.0]
print("lambda 2: ", lamda_2)
S_a_az_2 = [1.1, 1.1, 1.1, 1.1]
print("Sigma a az 2 = ", S_a_az_2[0])
S_a_otr_2 = [0.00124, 0.00124, 0.00124, 0.00124]
print("Digma a otr 2 = ", S_a_otr_2[0])
S_d_az_1 = [0.0306, 0.0306, 0.0306, 0.0306]
print("Sigma d az 1 = ", S_d_az_1[0])
S_d_otr_1 = [0.015, 0.015, 0.015, 0.015]
print("Sigma d otr 1 = ", S_d_otr_1[0])
nyS_f_az_1 = [0.0405, 0.0405, 0.0405, 0.0405]
print("ny * Sigma f az 1 = ", nyS_f_az_1 [0])
nyS_f_otr_1 = [0.0, 0.0, 0.0, 0.0]
print("ny * Sigma f otr 2 = ", nyS_f_otr_1[0])
nyS_f_az_2 = [2.2, 2.2, 2.2, 2.2]
print("ny * Sigma f az 2 = ", nyS_f_az_2[0])
iter = int(input())

# chisl
K = np.zeros(iter)
for count in range(0, iter):
  print('\nITERATION: ', count+1)

  #1 fast neutron
  print("\nТаблица 1:")
  gamma_az_1 = np.zeros(l//2)  # reshenia po formulam stogova
  for i in range(0, l//2 - 1):
    gamma_az_1[i] = d_r_1[i] / D_az_1[i]
  gamma_az_1[l//2-1] = d_r_1[l//2-1] / (2 * R_az) * (r_1[l//2-1+1] / D_az_1[l//2-1] + r_1[l//2-1+1] / D_otr_1[l//2-1])
  gamma_otr_1 = np.zeros(l//2)
  for j in range(0, l//2):
    gamma_otr_1[j] = d_r_1[j] / D_otr_1[j]
  gamma_1 = np.zeros(l)
  for z in range(0, l//2):
    gamma_1[z] = gamma_az_1[z]
  for y in range(l//2, l):
    gamma_1[y] = gamma_otr_1[y-l//2]
  print('gamma1: ', gamma_1)
  B_1 = np.zeros(l)
  B_1[0] = 0
  for t in range(1, l):
    B_1[t] = gamma_1[t] / gamma_1[t-1]
  print('B1: ', B_1)
  A_az_1 = np.zeros(l//2)
  for p in range(0, l//2):
    A_az_1[p] = r_1[p+1] / r_1[p] + B_1[p] * (r_1[p-1] / r_1[p]) + lamda_1[p] * gamma_1[p] * S_ad_az_1[p] * d_r_1[p]
  A_otr_1 = np.zeros(l//2)
  for m in range(0, l//2):
    A_otr_1[m] = r_1[m+1+l//2] / r_1[m+l//2] + B_1[m+l//2] * (r_1[m-1+l//2] / r_1[m+l//2]) + lamda_1[m+l//2] * gamma_1[m+l//2] * S_ad_otr_1[m] * d_r_1[m]
  A_1 = np.zeros(l)
  for e in range(0, l//2):
    A_1[e] = A_az_1[e]
  for c in range(l//2, l):
    A_1[c] = A_otr_1[c-l//2]
  print('A1: ', A_1)
  alfa_1 = np.zeros(l)
  alfa_1[0] = A_1[0]
  for w in range(1, l):
    alfa_1[w] = A_1[w] - B_1[w] / alfa_1[w-1]
  print('alfa1: ', alfa_1)
  Q_1 = np.zeros(l)
  for b in range(0, l//2):
    Q_1[b] = Q_az_1[b]
  for a in range(l//2, l):
    Q_1[a] = Q_otr_1[a-l//2]
  print('Q1: ', Q_1)
  f_1 = np.zeros(l)
  for s in range(l):
    f_1[s] = lamda_1[s] * gamma_1[s] * r_1[s] * d_r_1[s-1] * Q_1[s]
  print('f1: ', f_1)
  betta_1 = np.zeros(l)
  for v in range(l):
    betta_1[v] = f_1[v] + B_1[v] * betta_1[v-1] / alfa_1[v-1]
  print('betta1: ', betta_1)
  I_1 = np.zeros(l)
  I_1[l-1] = 0
  for d in range(l-2, -1, -1):
    I_1[d] = (I_1[d+1] + betta_1[d]) / alfa_1[d]
  print('I1: ', I_1)
  F_1 = np.zeros(l)
  for o in range(l):
    F_1[o] = I_1[o] / r_1[o]
  print('F1: ', F_1)

  #2 thermal neutron
  print("\nТаблица 2:")
  gamma_az_2 = np.zeros(l//2)
  for i in range(0, l//2 - 1):
    gamma_az_2[i] = d_r_2[i] / D_az_2[i]
  gamma_az_2[l//2-1] = d_r_2[l//2-1] / (2 * R_az) * (r_2[l//2-1+1] / D_az_2[l//2-1] + r_2[l//2-1+1] / D_otr_2[l//2-1])
  gamma_otr_2 = np.zeros(l//2)
  for j in range(0, l//2):
    gamma_otr_2[j] = d_r_2[j] / D_otr_2[j]
  gamma_2 = np.zeros(l)
  for z in range(0, l//2):
    gamma_2[z] = gamma_az_2[z]
  for y in range(l//2, l):
    gamma_2[y] = gamma_otr_2[y-l//2]
  print('gamma2: ', gamma_2)
  B_2 = np.zeros(l)
  B_2[0] = 0
  for t in range(1, l):
    B_2[t] = gamma_2[t] / gamma_2[t-1]
  A_az_2 = np.zeros(l//2)
  for p in range(0, l//2):
    A_az_2[p] = r_2[p+1] / r_2[p] + B_2[p] * (r_2[p-1] / r_2[p]) + lamda_2[p] * gamma_2[p] * S_a_az_2[p] * d_r_2[p]
  A_otr_2 = np.zeros(l//2)
  for m in range(0, l//2):
    A_otr_2[m] = r_2[m+1+l//2] / r_2[m+l//2] + B_2[m+l//2] * (r_2[m-1+l//2] / r_2[m+l//2]) + lamda_2[m+l//2] * gamma_2[m+l//2] * S_a_otr_2[m] * d_r_2[m]
  A_2 = np.zeros(l)
  for e in range(0, l//2):
    A_2[e] = A_az_2[e]
  for c in range(l//2, l):
    A_2[c] = A_otr_2[c-l//2]
  print('A2: ', A_2)
  alfa_2 = np.zeros(l)
  alfa_2[0] = A_2[0]
  for w in range(1, l):
    alfa_2[w] = A_2[w] - B_2[w] / alfa_2[w-1]
  print('alfa2: ', alfa_2)
  F_az_1 = np.zeros(l//2)
  for i in range(0, l//2):
    F_az_1[i] = F_1[i] ####
  F_otr_1 = np.zeros(l//2)
  for i in range(0, l//2):
    F_otr_1[i] = F_1[i+l//2]
  Q_az_2 = np.zeros(l//2)
  for u in range(0, l//2):
    Q_az_2[u] = S_d_az_1[u] * F_az_1[u]
  Q_otr_2 = np.zeros(l//2)
  for g in range(0, l//2):
    Q_otr_2[g] = S_d_otr_1[g] * F_otr_1[g]
  Q_2 = np.zeros(l)
  for b in range(0, l//2):
    Q_2[b] = Q_az_2[b]
  for a in range(l//2, l):
    Q_2[a] = Q_otr_2[a-l//2]
  print('Q2: ', Q_2)
  f_2 = np.zeros(l)
  for s in range(l):
    f_2[s] = lamda_2[s] * gamma_2[s] * r_2[s] * d_r_2[s-1] * Q_2[s]
  print('f2: ', f_2)
  betta_2 = np.zeros(l)
  for v in range(l):
    betta_2[v] = f_2[v] + B_2[v] * betta_2[v-1] / alfa_2[v-1]
  print('betta2: ', betta_2)
  I_2 = np.zeros(l)
  I_2[l-1] = 0
  for d in range(l-2, -1, -1):
    I_2[d] = (I_2[d+1] + betta_2[d]) / alfa_2[d]
  print('I2: ', I_2)
  F_2 = np.zeros(l)
  for o in range(l):
    F_2[o] = I_2[o] / r_2[o]
  print('F2: ', F_2)

  #3 K besk
  print("\nТаблица 3:")
  print('l = ', l//2)
  r_az = np.zeros(l//2)
  for i in range(0, l//2):
    r_az[i] = r_1[i]
  N_az_1 = nyS_f_az_1 * F_az_1
  print('ny*Sigma*F1: ', N_az_1)
  N_otr_1 = nyS_f_otr_1 * F_otr_1
  F_az_2 = np.zeros(l//2)
  for j in range(0, l//2):
    F_az_2[j] = F_2[j]
  N_az_2 = nyS_f_az_2 * F_az_2
  print('ny*Sigma*F2: ', N_az_2)
  q_b_1 = np.zeros(l//2)
  for u in range(0, l//2):
    q_b_1[u] = nyS_f_az_1[u] * F_1[u] + nyS_f_az_2[u] * F_2[u]
  print('q b 1: ', q_b_1)
  d_V = np.zeros(l//2)
  d_r_3 = 0.5 * (r_az[1] - r_az[0])
  for k in range(0, l//2):
    d_V[k] = r_az[k]**2 * d_r_3
  print('delta V: ', d_V)
  M = q_b_1 * d_V
  print('delta V * q1: ', M)
  symm_qV_n = 0
  for n in range(0, l//2):
    symm_qV_n += q_b_1[n] * d_V[n]
  symm_V = 0
  for m in range(0, l//2):
    symm_V += d_V[m] * Q_1[m]
  print('F1: ', F_1)
  Q_az_1 = q_b_1
  K_eff_az = symm_qV_n / symm_V
  print('K eff = ', K_eff_az)
  K[count] = K_eff_az
print('K0 = ', K[0])
print('avg K: ', np.average(K))

# dop
r1 = np.zeros(l+1)
for i in range(l+1):
  r1[i] = r_1[i] * (-1)
F1 = np.append(F_1, 0)
F2 = np.append(F_2, 0)
x, xx = [16, 16], [-16, -16]
x1, xx1 = [30, 30], [-30, -30]
y = np.linspace(0, max(F_1), 2)

# grafik
plt.plot(r_1, F1, label="Ф-1", color = 'blue')
plt.plot(r1, F1, color = 'blue')
plt.plot(r_1, F2, color = 'red')
plt.plot(r1, F2, label="Ф-2", color = 'red')
plt.plot(x, y, label="Граница аз-отр", color = 'green')
plt.plot(xx, y, color = 'green')
plt.plot(x1, y, label="Граница р-вак", color = 'black')
plt.plot(xx1, y, color = 'black')
plt.xlabel('r, см')
plt.ylabel('Ф, нейтр/см^2*с')
plt.title('График распределения нейтронов по реактору')
plt.legend()
plt.grid()
plt.show()
