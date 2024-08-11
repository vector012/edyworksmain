a = int(input())
b = int(input())
c = input()
if c == '+' or c == '-' or c == '*' or c == '/':
    if c == '+':
        print(a + b)
    elif c == '-':
        print(a - b)
    elif c == '*':
        print(a * b)
    elif c == '/':
        if b != 0:
            print(a / b)
        else:
            print('На ноль делить нельзя!')
else:
    print('Неверная операция')
