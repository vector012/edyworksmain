# bibl

import numpy as np
from matplotlib import pyplot as plt

#yslov

print('6.14 Приближённое решение краевой задачи для обыкновенного'
      ' дифференциального уравнения \n'
      'Найти приближённое решение краевой задачи для обыкновенного '
      'дифференциального уравнения на отрезке [0,1] с шагом h. \nДля вычисления'
      'решения использовать метод прогонки с краевыми условиями первого и '
      'второго порядка точности. \nДля сравнения приведено точное решение u0(x): \n'
    '''u" + (cosx)u' + (sinx)u = xsinx, \n'''
      "3u(0) - u'(0) = 2, 2u(1) + u'(1) = 3.2391, \n"
      'u0(x) = x + cosx \nРезультат:')

# kod funk

class zadacha():

  # задание условий задачи 6.14
  # y'' + p*y' + q*y = F
  # alfa_0*y + alfa_1*y' = gran_1
  # betta_0*y + betta_1*y' = gran_2
  # Anatit = f

  def Thomas_alg(self, alfa_0, alfa_1, betta_0, betta_1, gran_1, gran_2, a, b):

  #3

    j = 5
    M = np.zeros(j)
    M[0] = 10
    for index0 in range(1, len(M)):
      M[index0] = M[index0-1] * 2
    otc_1 = np.zeros(len(M))
    otc_2 = np.zeros(len(M))
    H = np.zeros(len(M))
    for index1 in range(len(M)):

  #0

      N = int(M[index1])
      self.alfa_0 = alfa_0
      self.alfa_1 = alfa_1
      if np.abs(alfa_0) + np.abs(alfa_1) == 0:
        print('bug gr1')
      self.betta_0 = betta_0
      self.betta_1 = betta_1
      if np.abs(betta_0) + np.abs(betta_1) == 0:
        print('bug gr2')
      self.gran_1 = gran_1
      self.gran_2 = gran_2
      if np.abs(gran_1) == 0 and np.abs(gran_2) == 0:
        print('bug gr_chis')
      self.a = a
      self.b = b
      if N == 0 or b == 0:
        print('bug oblast')

  #1

      x = np.linspace(a, b, N)
      Analit = x + np.cos(x)
      F = x * np.sin(x)
      p = np.cos(x)
      q = np.sin(x)
      h = (b - a) / (N - 1)

      # y'[i] = (y[i+1] - y[i-1])/2*h
      # A*y[i+1] + B*y[i] + C*y[i-1] = F

      A = 1 / (h ** 2) - p / (2 * h)
      A[N-1] = -betta_1 / h
      B = (-2) / (h ** 2) + q
      B[0] = alfa_0 - alfa_1 / h
      B[N-1] = betta_1 / h + betta_0
      C = 1 / (h ** 2) + p / (2 * h)
      C[0] = alfa_1 / h
      for index2 in range(N):
        if A[index2] == 0 or B[index2] == 0 or C[index2] == 0:
          print('bug coef', index2)
      F[0] = gran_1
      F[N-1] = gran_2

      I_1 = np.zeros(N)
      I_1[0] = F[0] / B[0]
      I_2 = np.zeros(N)
      I_2[0] = -C[0] / B[0]
      for index2 in range(1, N):
        I_1[index2] = (F[index2] - A[index2] * I_1[index2-1]) \
        / (B[index2] + A[index2] * I_2[index2-1])
        I_2[index2] = -C[index2] / (B[index2] + A[index2] * I_2[index2-1])
      for index3 in range(N):
        if I_1[index3] == 0 or I_2[index3] == 0:
          print('bug count', index3)
      I_2[N-1] = 0
      y = np.zeros(N)
      y[N-1] = I_1[N-1]
      for index4 in range(N-2, 0, -1):
        y[index4] = I_1[index4] + I_2[index4] * y[index4+1]
      y_1 = y

      if N == 20:
        plt.title('График №1. Зависимость решения от координаты #≈↗')
        plt.plot(x[1::], Analit[1::], label = 'Точное решение', color = 'black' \
                , linewidth = 3)
        plt.plot(x[1::], y[1::], label = 'Ⅰ порядок приб. точ.', color = 'red')

  #2

      x = np.linspace(a, b, N)
      Analit = x + np.cos(x)
      F = x * np.sin(x)
      p = np.cos(x)
      q = np.sin(x)
      h = (b - a) / (N - 1)
      A = 1 / (h ** 2) - p / (2 * h)
      A[N-1] = 2 / h ** 2
      B = (-2) / (h ** 2) + q
      B[0] = -2 / h ** 2 + 2 * alfa_0 / (alfa_1 * h) - \
            p[0] * alfa_0 / alfa_1 + q[0]

      B[N-1] = -2 * betta_0 / (betta_1 * h) - 2 / h ** 2 - \
              p[N-1] * betta_0 / betta_1 + q[N-1]
      C = 1 / (h ** 2) + p / (2 * h)
      C[0] = 2 / h ** 2
      f = np.zeros(N)
      f = F
      f[0] = F[0] + 2 * gran_1 / (alfa_1 * h) - gran_1 * p[0] / alfa_1
      f[N-1] = F[N-1] - 2 * gran_2 / (betta_1 * h) - gran_2 * p[N-1] / betta_1

      I_1 = np.zeros(N)
      I_1[0] = F[0] / B[0]
      I_2 = np.zeros(N)
      I_2[0] = -C[0] / B[0]
      for index5 in range(1, N):
        I_1[index5] = (F[index5] - A[index5] * I_1[index5-1]) \
        / (B[index5] + A[index5] * I_2[index5-1])
        I_2[index5] = -C[index5] / (B[index5] + A[index5] * I_2[index5-1])
      for index6 in range(N):
        if I_1[index6] == 0 or I_2[index6] == 0:
          print('bug count', index6)
      I_2[N-1] = 0
      y = np.zeros(N)
      y[N-1] = I_1[N-1]
      for index7 in range(N-2, 0, -1):
        y[index7] = I_1[index7] + I_2[index7] * y[index7+1]
      y_2 = y

      if N == 20:
        plt.plot(x[1::], y[1::], label = 'Ⅱ+ порядок приб. точ.', \
                 color = 'green', linestyle = '--')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.show()

  #3

      H[index1] = h
      for index8 in range(N):
        D1 = np.abs(y_1[index8] - Analit[index8])
        if D1 > otc_1[index1] and index8 > 0:
          otc_1[index1] = D1
        D2 = np.abs(y_2[index8] - Analit[index8])
        if D2 > otc_2[index1] and index8 > 0:
          otc_2[index1] = D2

    plt.title('График №2. Зависимости отклонения от шага #△↷')
    plt.plot(np.log(H), np.log(otc_1), label = 'max отклонение 1 порядка', color = 'red')
    plt.plot(np.log(H), np.log(otc_2), label = 'max отклонение 2 порядка', color = 'green')
    plt.xlabel('h')
    plt.ylabel('delta')
    plt.legend()
    plt.grid()
    plt.show()

    # y = k*x + b

    k_1 = np.zeros(len(M)//2)
    for index10 in range(0, len(M) // 2):
      k_1[index10] = np.abs( (np.log(otc_1[index10]) - np.log(otc_1[-index10-1]) \
                            / (np.log(H[index10]) - np.log(H[-index10-1]))) )
    k_2 = np.zeros(len(M)//2)
    for index11 in range(0, len(M) // 2):
      k_2[index11] = np.abs( (np.log(otc_2[index11]) - np.log(otc_2[-index11-1]) \
                            / (np.log(H[index11]) - np.log(H[-index11-1]))) )

    k1 = np.mean(k_1)
    print('tg⊾_1 -> O(1). k1 = ', np.round(float(k1), 1))
    k2 = np.mean(k_2)
    print('tg⊾_2 -> O(2). k2 = ', np.round(float(k2), 1))
    if k1 == 0 or k2 == 0:
      print('bug k')
    # k_1 = abs((np.log(otc_1[0])-np.log(otc_1[-1])/(np.log(H[0])-np.log(H[-1]))))
    # k_2 = abs((np.log(otc_2[0])-np.log(otc_2[-1])/(np.log(H[0])-np.log(H[-1]))))
    # print("k_1 = ", k_1)
    # print("k_2 = ", k_2)

  # start

zadacha1 = zadacha()
zadacha1.Thomas_alg(3, -1, 2, 1, 2, 3.2391, 0, 1)

# Краткое описание: задаю класс краевых задач, т.к. все они похожи, меняются
# только коэффициенты. Внутри  класса создаю массив для числа узлов, шагов.
# Задаю массив для отклонений их столько же, сколько шагов. Дальше цикл - для
# каждого шага(для каждого N) сначала определяю коэффициенты и проверяю, что
# введённые коэфы подходят. Потом в 1ом пункте пишу в явном виде МОИ гр усл и
# неоднородность уравнения, т.к. не знаю, как их передать в класс не в виде
# строки. Потом через кон разностные схемки привожу ОДУ 2го порядка к у в точке
# слева и справа. Нахожу коэфы при этих у. I_1 и 2 это прогоночные коэфы. Через
# них и находится у. В пункте 2 проделываю тоже самое для коэфов вычисленных
# с большей точностью. Строю графики. В пункте 3 расчёт отклонений - забиваю
# мссив шагов шагами на данном этапе общего цикла и забиваю отклонения:
# если разность численного и аналитического реш больше предыдущей, то она
# перезаписывается. Строю график. рассчитываю тангенсы угла наклона, чтобы
# понять какой порядок точности. В самом конце кода я вызываю класс, забивая
# его моими константами из задачи.
