import numpy as np
from matplotlib import pyplot as plt


T = ['cross.txt']
E = ['energ.txt']
res = open('res_new.txt', 'w')
res.close()
for i1 in range(len(T)):
    inf = open(T[i1], 'r', encoding='UTF-8')
    I = np.array([])
    inp = inf.readlines()
    inf.close()
    ener = open(E[i1], 'r', encoding='UTF-8')
    I2 = np.array([])
    inp2 = ener.readlines()
    ener.close()
    for j1 in range(len(inp)):
        inp1 = str(inp[j1])
        I = np.append(I, float(inp1))
        inp21 = str(inp2[j1])
        I2 = np.append(I2, float(inp21))
    res = open('res_new.txt', 'a+', encoding='UTF-8')
    X3 = np.array([])
    Y3 = np.array([])

    I3 = np.copy(I)
    for j2 in range(len(I)):
        i = str(I[j2])
        d = np.abs(len(i) - i.find('.') - 1)
        I[j2] = round(I[j2])

    res.write('\nТест 3\n')
    evo3 = 0
    for k3 in range(241, len(I3)):
        r = float(I3[k3])
        t = float(I2[k3])
        x = (round(r+0.1)) % (round(np.log(t+0.1)))
        if x == 0:
            a = str(k3)
            b = str(I3[k3])
            c = str(I2[k3])
            res.write(f'{a} значение {b} кратно логарифму энергии {c}\n')
            X3 = np.append(X3, float(I2[k3]))
            Y3 = np.append(Y3, float(I3[k3]))
            evo3 += 1
    print(evo3)
    '''
    Nmin = 241 Group = 1 Emin = 1.554743
    Nmin = 432 Group = 2 Emin = 4.423955
    Nmin = 970 Group = 3 Emin = 12.09827
    Nmin = 2161 Group = 4 Emin = 33.01962
    Nmin = 4501 Group = 5 Emin = 89.92931
    Nmin = 9385 Group = 6 Emin = 244.6512
    Nmin = 17852 Group = 7 Emin = 665.051
    Nmin = 33781 Group = 8 Emin = 1808.024
    Nmin = 39480 Group = 9 Emin = 4996.802
    Nmin = 39613 Group = 10 Emin = 13687.03
    Nmin = 39849 Group = 11 Emin = 36340.0
    Nmin = 40135 Group = 12 Emin = 100000.0
    Nmin = 40161 Group = 13 Emin = 270000.0
    Nmin = 40207 Group = 14 Emin = 750000.0
    Nmin = 40229 Group = 15 Emin = 2000000.0
    Nmin = 40251 Group = 16 Emin = 5500000.0
    Nmin = 40284 Group = 17 Emin = 15000000.0
    '''
    X = 1
    J = np.array([]) # Group
    N = np.array([]) # Nmin
    O = np.array([]) # Emin
    aver = np.array([])
    for kk in range(len(I2)):
        if round(np.log(I2[kk]+0.1)) >= X:
            N = np.append(N, kk)
            J = np.append(J, X-1)
            O = np.append(O, I2[kk])
            X += 1
    N = np.append(N, int(40305))
    J = np.append(J, X-1)
    O = np.append(O, int(3.0e7))
    for s in range(len(J)-1):
        m = I3[int(N[s]):int(N[s+1]):]
        aver = np.append(aver, np.mean(m / (s+1)))
    print(aver)
    gg = 0
    '''otcl = np.array([])
    for kk2 in range(int(N[0]), int(N[16])):
        if I2[kk2] <= O[gg]:
            otcl = np.append(otcl, ((I3[kk2] - np.log(I2[kk2]) * aver[gg]) / I3[kk2] * 100))
        else:
            gg += 1
    print(otcl)'''
    '''div = np.array([])
    NN = np.array([0])
    NN = np.append(NN, N)
    for kk3 in range(len(NN)-1):
        reas = otcl[int(NN[kk3]):int(NN[kk3+1]):]
        div = np.append(div, np.mean(reas))
    print(div)'''

    B1,B2,B3,B4,B5,B6,B7 = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    B8,B9,B10,B11,B12,B13 = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    B14,B15,B16,B17 = np.array([]),np.array([]),np.array([]),np.array([])
    D = np.array([])
    for i in range(int(N[0]), int(N[1])):
        B1 = np.append(B1, (I3[i] - np.log(I2[i]) * aver[0]) / I3[i] * 100)
        #C = np.append(C, np.mean(B[0:int(N[1] - N[0]):]))
    D = np.append(D, np.mean(B1))
    for i in range(int(N[1]), int(N[2])):
        B2 = np.append(B2, (I3[i] - np.log(I2[i]) * aver[1]) / I3[i] * 100)
    D = np.append(D, np.mean(B2))
    for i in range(int(N[2]), int(N[3])):
        B3 = np.append(B3, (I3[i] - np.log(I2[i]) * aver[2]) / I3[i] * 100)
    D = np.append(D, np.mean(B3))
    for i in range(int(N[3]), int(N[4])):
        B4 = np.append(B4, (I3[i] - np.log(I2[i]) * aver[3]) / I3[i] * 100)
    D = np.append(D, np.mean(B4))
    for i in range(int(N[4]), int(N[5])):
        B5 = np.append(B5, (I3[i] - np.log(I2[i]) * aver[4]) / I3[i] * 100)
    D = np.append(D, np.mean(B5))
    print(B5)
    for i in range(int(N[5]), int(N[6])):
        B6 = np.append(B6, (I3[i] - np.log(I2[i]) * aver[5]) / I3[i] * 100)
    D = np.append(D, np.mean(B6))
    for i in range(int(N[6]), int(N[7])):
        B7 = np.append(B7, (I3[i] - np.log(I2[i]) * aver[6]) / I3[i] * 100)
    D = np.append(D, np.mean(B7))
    for i in range(int(N[7]), int(N[8])):
        B8 = np.append(B8, (I3[i] - np.log(I2[i]) * aver[7]) / I3[i] * 100)
    D = np.append(D, np.mean(B8))
    for i in range(int(N[8]), int(N[9])):
        B9 = np.append(B9, (I3[i] - np.log(I2[i]) * aver[8]) / I3[i] * 100)
    D = np.append(D, np.mean(B9))
    for i in range(int(N[9]), int(N[10])):
        B10 = np.append(B10, (I3[i] - np.log(I2[i]) * aver[9]) / I3[i] * 100)
    D = np.append(D, np.mean(B10))
    for i in range(int(N[10]), int(N[11])):
        B11 = np.append(B11, (I3[i] - np.log(I2[i]) * aver[10]) / I3[i] * 100)
    D = np.append(D, np.mean(B11))
    for i in range(int(N[11]), int(N[12])):
        B12 = np.append(B12, (I3[i] - np.log(I2[i]) * aver[11]) / I3[i] * 100)
    D = np.append(D, np.mean(B12))
    for i in range(int(N[12]), int(N[13])):
        B13 = np.append(B13, (I3[i] - np.log(I2[i]) * aver[12]) / I3[i] * 100)
    D = np.append(D, np.mean(B13))
    for i in range(int(N[13]), int(N[14])):
        B14 = np.append(B14, (I3[i] - np.log(I2[i]) * aver[13]) / I3[i] * 100)
    D = np.append(D, np.mean(B14))
    for i in range(int(N[14]), int(N[15])):
        B15 = np.append(B15, (I3[i] - np.log(I2[i]) * aver[14]) / I3[i] * 100)
    D = np.append(D, np.mean(B15))
    for i in range(int(N[15]), int(N[16])):
        B16 = np.append(B16, (I3[i] - np.log(I2[i]) * aver[15]) / I3[i] * 100)
    D = np.append(D, np.mean(B16))
    for i in range(int(N[16]), int(N[17])):
        B17 = np.append(B17, (I3[i] - np.log(I2[i]) * aver[16]) / I3[i] * 100)
    D = np.append(D, np.mean(B17))
    print()
    print(len(N))
    print(N[17])
    print(len(aver))
    print(aver[16])
    print()
    print(D)

    '''[50.01499628 61.91652015 33.86824707 18.24554066  9.03178066  5.13570822
    3.34737456  2.55105139  1.82212232  1.4137905   1.19757148  0.9091825
    0.66920109  0.50004127  0.51435478  0.39463787  0.35211243]'''

    X2 = np.array([])
    Y2 = np.array([])
    res.write('\nТест 2\n')
    evo2 = 0
    for k2 in range(int(N[0])):
        r = float(I3[k2])
        t = float(I2[k2])
        x = (round(r+0.1)) % (round(np.log((1/t+0.1))))
        if x == 0:
            a = str(k2)
            b = str(I3[k2])
            c = str(I2[k2])
            res.write(f'{a} значение {b} кратно логарифму энергии {c}\n')
            X2 = np.append(X2, float(I2[k2]))
            Y2 = np.append(Y2, float(I3[k2]))
            evo2 += 1
    print(evo2)

    X2 = 1
    J2 = np.array([]) # Group
    N2 = np.array([]) # Nmin
    O2 = np.array([]) # Emin
    aver2 = np.array([])
    for k2k in range(len(I2)):
        if round(np.log((1/I2[k2k]+0.1))) >= X:
            N2 = np.append(N, k2k)
            J2 = np.append(J, X-1)
            O2 = np.append(O, I2[k2k])
            X2 += 1
    for s2 in range(len(J)-1):
        m = I3[int(N[s2]):int(N[s2+1]):]
        aver2 = np.append(aver, np.mean(m / (s2+1)))
    print(aver2)

    res.close()

    plt.plot(np.log(X3), np.log(Y3),  label = 'кратно log E')
    plt.scatter(np.log(X3), np.log(Y3), color="orange")
    plt.xlabel('E')
    plt.ylabel('sigma')
    plt.legend()
    plt.show()
