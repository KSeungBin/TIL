# 1874. 스택 수열
# 문제 유형 - 스택, 그리디
# 1. 스택에 원소를 삽입할 때, 단순히 특정 수에 도달할 떄까지 삽입하면 된다
# 2. 스택에서 원소를 연달아 빠낼 때 내림차순을 유지할 수 있는지 확인

n = int(input())

count = 1
stack = []
result = []

for i in range(1, n+1):
    data = int(input())
    while count <= data:
        stack.append(count)
        count += 1
        result.append('+')
        
    if stack[-1] == data:
        stack.pop()
        result.append('-')
    else:
        print('NO')
        exit(0)

print('\n'.join(result))