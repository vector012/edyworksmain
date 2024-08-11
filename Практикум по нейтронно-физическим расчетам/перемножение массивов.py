import numpy as np

def input_data(array, rows):  # input in array
    data = 0
    for i in range(rows):
        data = input().split()
        for j in range(len(data)):
            data[j] = int(data[j])
            array[i][j] += data[j]
    return array
def out_data(array, rows, rows_for_print):  # print array as rows
    if rows_for_print <= rows:
        for i in range(rows_for_print):
            print(array[i])
    else:
        print('!INPUT  ERROR!')
def multiplication(array_a, array_b, row1, column1, row2, column2):  # multiplication of arrays
    array_c = 0
    if column1 == 1 and column1 == row2:
        array_c = np.zeros((row1, column2))
        for i in range(row2):
            for j in range(column2):
                array_c += array_a[i] * array_b[i][j]
    elif column2 == 1 and column1 == row2 != 1:
        for i in range(row1):
            for j in range(column1):
                array_c += array_a[i][j] * array_b[i]
    elif column1 == row2 == 1:
        for i in range(row1):
            for j in range(column1):
                array_c += array_a[i] * array_b[j]
    elif column1 == row2 != 1 and row1 == column2 == 1:
        for i in range(row1):
            for j in range(column1):
                array_c += array_a[i] * array_b[j]
    elif column1 == row2 != 1:
        array_c = np.zeros((row1, column2))
        for i in range(row1):
            for j in range(column2):
                for t in range(column1):
                    array_c[i][j] += array_a[i][t] * array_b[t][j]
    return array_c
def out_data_multiplication(array, rows_for_print):  # print array as rows
    for i in range(rows_for_print):
        print(array[i])
def transposition(array_a, row, column):  # transposition of array
    array_b = np.zeros((row, column))
    if row == column != 1:
        for i in range(row):
            for j in range(column):
                array_b[j][i] = array_a[i][j]
        return array_b

print('No.3 The program takes two arrays line by line and outputs their multiplication line by line. also by transplanting.')
r1 = int(input('Type number of rows of 1st array: '))
c1 = int(input('Type number of columns of 1st array: '))
if r1 > 0 and c1 > 0:  # logic condition
    a = np.zeros((r1, c1))  # arr1
else:
    print('!INPUT  ERROR!\n')
r2 = int(input('Type number of rows of 2nd array: '))
c2 = int(input('Type number of columns of 2nd array: '))
if r2 > 0 and c2 > 0:
    b = np.zeros((r2, c2))  # arr2
else:
    print('!INPUT  ERROR!\n')
print('Type str data for array 1.\n'
      ' Столько символов через пробел, сколько в нём столбцов, '
      'далее энтер до тех пор, пока не заполнятся все строки')
a = input_data(a, r1)  # run input
print('Type str data for array 2.\n'
      ' Столько символов через пробел, сколько в нём столбцов, '
      'далее энтер до тех пор, пока не заполнятся все строки')
b = input_data(b, r2)
g = multiplication(a, b, r1, c1, r2, c2)  # remember mult result as new arr
r3_o = int(input('How much rows of result array do you want to see: '))
print('Multiplication result: ')
out_data_multiplication(g, r3_o)
print('For comparison: \n', a @ b)
print('Transposition result: ')
print(transposition(a, r1, c1))  # run transp
print(transposition(b, r2, c2))
print('For comparison: \n', a.T, '\n', b.T)
