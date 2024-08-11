# библиотека для работы с массивами и математикой
import numpy as np
# библиотека для создания диаграмм
from matplotlib import pyplot as plt

# база данных Incident neutron data_Reaction cross sections_MT
S = ['IND_RCS_t.txt']#, 'IND_RCS_s.txt', 'IND_RCS_a.txt']
# база данных Incident neutron data_Reaction energy_MT
E = ['IND_RE_t.txt']#, 'IND_RE_s.txt', 'IND_RE_a.txt']
# создание файла выходных данных
res = open('calculation result.txt', 'w')
res.close()
for i1 in range(len(S)):
    # Чтение файлов
    sect = open(S[i1], 'r', encoding='UTF-8')
    IS = np.array([])
    inpus = sect.readlines()
    sect.close()
    energ = open(E[i1], 'r', encoding='UTF-8')
    IE = np.array([])
    inpue = energ.readlines()
    energ.close()
    for j1 in range(len(inpus)):
        inp1 = str(inpus[j1])
        IS = np.append(IS, float(inp1))
        inp2 = str(inpue[j1])
        IE = np.append(IE, float(inp2))
    res = open('calculation result.txt', 'a+', encoding='UTF-8')
    # заполнение массива округленными дробными чисами
    IS_mode = np.array([])
    for j2 in range(len(IS)):
        data = str(IS[j2])
        leng = np.abs(len(data) - data.find('.') - 1)
        IS_mode = np.append(IS_mode, np.round((IS[j2] + 0.001), 1))
        '''# точность - до целых!
        IS_mode = np.append(IS_mode, np.round(IS[j2]))'''
    # сетки графиков тепловой, резонансной и быстрой областей
    Xt = np.array([])
    Yt = np.array([])
    Xr = np.array([])
    Yr = np.array([])
    Xf = np.array([])
    Yf = np.array([])
    n = str(S[i1])
    res.write(f'    Данные файла {n}\n')
    # отбор сечений по области энергии (быстрой)
    X = 5
    Gf = np.array([])
    Nf = np.array([])
    Ef = np.array([])
    for k1 in range(len(IE)):
        if (10 * np.round((np.log(IE[k1]) + 0.001), 1)) >= X:
            Gf = np.append(Gf, int((X - 5) / 10))
            Nf = np.append(Nf, int(k1))
            Ef = np.append(Ef, float(IE[k1]))
            X += 10
    Gf = np.append(Gf, int((X - 5) / 10))
    Nf = np.append(Nf, len(IE) - 1)
    Ef = np.append(Ef, float(IE[-1]))
    print('\nGf -', Gf, '\nNf -', Nf, '\nEf -', Ef, '\n')
    '''# точность - до целых!
    X = 1
    Gf = np.array([])
    Nf = np.array([])
    Ef = np.array([])
    for k1 in range(len(IE)):
        if np.round(np.log(IE[k1])) >= X:
            Gf = np.append(Gf, X - 1)
            Nf = np.append(Nf, k1)
            Ef = np.append(Ef, IE[k1])
            X += 1
    Gf = np.append(Gf, X - 1)
    Nf = np.append(Nf, len(IE))
    Ef = np.append(Ef, IE[-1])
    print('Gf -', Gf, 'Nf -', Nf, 'Ef -', Ef)'''
    # проверка кратности сечений
    res.write('ТЕСТ 1 для БО\n')
    elo1 = 0
    NULL1 = np.array([])
    for k2 in range(int(Nf[0]), len(IS)):
        x = (10 * IS_mode[k2]) % (10 * np.round((np.log(IE[k2]) + 0.001), 1))
        # + 0.001 т.к. 5.45 - 5.4, а 5.451 - 5.5, 5.4515 аналогично
        '''1. проблема в том, Что когда под логарифмом около единицы, т.е. энергия от 0.999 до 1.001
        питон превращает логарифм в 0
        2. в приницпе проблема деления на 0 после округления до 1 знака с последующим
        умножением на 10, т.е. энергии от 0.95 до 1.055 эВ'''
        # определять эти энергии надо как-то по равенству операции деления и null/nan, но проще вручную
        '''# точность - до целых!
        x = IS_mode[k2] % np.round(np.log(IE[k2]))'''
        if np.isnan(x):
            NULL1 = np.append(NULL1, k2)
        if x == 0:
            a = str(k2)
            b = str(IS[k2])
            c = str(IE[k2])
            res.write(f'№ {a} - значение сечения {b} кратно логарифму соответствующей ему энергии {c}\n')
            Xf = np.append(Xf, float(IE[k2]))
            Yf = np.append(Yf, float(IS[k2]))
            elo1 += 1
    e1 = elo1
    # отбор сечений по области энергии (тепловой)
    X2 = 5
    Gt = np.array([])
    Nt = np.array([])
    Et = np.array([])
    for k1t in range(len(IE) - 1, 0, -1):
        if (-10 * np.round((np.log(IE[k1t]) + 0.001), 1)) >= X2:
            Gt = np.append(Gt, int((X2 - 5) / 10))
            Nt = np.append(Nt, int(k1t))
            Et = np.append(Et, float(IE[k1t]))
            X2 += 10
    Gt = np.append(Gt, int((X2 - 5) / 10))
    Nt = np.append(Nt, 0)
    Et = np.append(Et, float(IE[0]))
    print('Gt -', Gt, '\nNt -', Nt, '\nEt -', Et, '\n')
    # проверка кратности сечений TO
    res.write('ТЕСТ 2 для ТО\n')
    elo2 = 0
    NULL2 = np.array([])
    for k2t in range(int(Nt[-1]), int(Nt[0]), -1):
        x2 = (10 * IS_mode[k2t]) % (-10 * np.round((np.log(IE[k2t]) + 0.001), 1))
        if np.isnan(x2):
            NULL2 = np.append(NULL2, k2t)
        if x2 == 0:
            a = str(k2t)
            b = str(IS[k2t])
            c = str(IE[k2t])
            res.write(f'№ {a} - значение сечения {b} кратно минус логарифму соответствующей ему энергии {c}\n')
            Xt = np.append(Xt, float(IE[k2t]))
            Yt = np.append(Yt, float(IS[k2t]))
            elo2 += 1
    e2 = elo2
    e = e1 + e2
    print(f'Истинно кратных сечений - {e} штук\n')
    # рассчёт констант БО
    aver1 = np.array([])
    for k3 in range(len(Gf) - 1):
        obl = IS_mode[int(Nf[k3]):int(Nf[k3 + 1]):]
        '''obl = IS[int(Nf[k3]):int(Nf[k3 + 1]):]'''
        aver1 = np.append(aver1, np.round(np.mean(obl / (k3 + 1)), 3))
    a1 = aver1
    print(f'Коэффициенты РО + БО\n{a1}\n')
    # рассчёт констант ТО
    aver2 = np.array([])
    for k3t in range(len(Gt) - 1):
        obl2 = IS_mode[int(Nt[k3t + 1]):int(Nt[k3t]):]
        aver2 = np.append(aver2, np.round(np.mean(obl2 / (k3t + 1)), 3))
    a2 = aver2
    print(f'Коэффициенты ТО\n{a2}\n')
    # Расчёт погрешности БО
    O = np.array([])
    D = np.array([])
    for k4 in range(len(Nf) - 1):
        for h1 in range(int(Nf[k4]), int(Nf[k4 + 1])):
            O = np.append(O, (IS[h1] - np.abs(np.log(IE[h1])) * aver1[k4]) / IS[h1] * 100)
            # мб не IS а IS_mode брать и IE тоже округлять?
            # abs т.к. в тепл обл ln отрицательный
        D = np.append(D, np.round(np.mean(O), 2))
        O = np.array([])
    d = D
    print(f'Отклонения РО + БО\n{d}\n')
    # Расчёт погрешности TО
    O2 = np.array([])
    D2 = np.array([])
    for k4t in range(len(Nt) - 2):
        # len(Nt) - 2 т.к. последняя группа из 1 элемента, там не может быть погрешности
        for h1t in range(int(Nt[k4t + 1]), int(Nt[k4t] - 1)):
            O2 = np.append(O2, (IS[h1t] - np.abs(np.log(IE[h1t])) * aver2[k4t]) / IS[h1t] * 100)
            # мб не IS а IS_mode брать и IE тоже округлять?
            # abs т.к. в тепл обл ln отрицательный
        D2 = np.append(D2, np.round(np.mean(O2), 2))
        O2 = np.array([])
    d2 = D2
    print(f'Отклонения TО\n{d2}')
    res.close()
    plt.plot(np.log(Xf), np.log(Yf))
    plt.scatter(np.log(Xf), np.log(Yf), color='orange', label='кратно ln E')
    plt.xlabel('E')
    plt.ylabel('sigma')
    plt.title('График зависимости сеченя от энергии в лог.масшт.')
    plt.legend()
    plt.show()
    '''plt.plot(np.log(Xt), np.log(Yt))
    plt.scatter(np.log(Xt), np.log(Yt), color='orange', label='кратно -ln E')
    plt.xlabel('E')
    plt.ylabel('sigma')
    plt.title('График зависимости сеченя от энергии в лог.масшт.')
    plt.legend()
    plt.show()'''
    # Пусто т.к. среди 190 сечения истинно кратных логарифму нет
