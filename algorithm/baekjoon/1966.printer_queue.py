# printer queue
# queue, implementation, greedy
# greedy : because N is smaller than 100, it doesn't matter current result will bring the overall optimal result.
# exit program when the largest number corresponds to M and placed at the beginning of the list

from ast import While


test_case = int(input())

for _ in range(test_case):
    n, m = list(map(int, input().split('')))
    queue = list(map(int, input().split('')))

    # TIP 1 : generate list whose elements are tuple including increasing index
    queue = [(i,index) for index, i in enumerate(queue)]

    count = 0
    while True:
        # TIP 2 : search and return the largest i
        if queue[0][0] == max(queue, key=lambda x: x[0])[0]:
            count += 1
            if queue[0][1] == m:
                print(count)
                break
            else:
                queue.pop()
        else:
            queue.append(queue.pop())