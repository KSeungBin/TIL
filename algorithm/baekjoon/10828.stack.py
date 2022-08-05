# time out : if you use "in" operator, it is faster to convert it to a set in advance


import sys
input=sys.stdin.readline

n = int(input())
stack = []

for i in range(n):
    data = input().strip() # readline() method work in python when removes any leading and/or trailing whitespaces from a string
    if len(data) >= 6:     # i.e. data = push 3
        number = int(data.split(" ")[1])
        stack.append(number)
    else: 
        if data == 'pop':
            [print(-1) if len(stack)==0 else print(stack.pop())]
        if data == 'size':
            print(len(stack))
        if data == 'empty':
            [print(1) if len(stack)==0 else print(0)]
        if data == 'top':
            [print(-1) if len(stack)==0 else print(stack[-1])]
    