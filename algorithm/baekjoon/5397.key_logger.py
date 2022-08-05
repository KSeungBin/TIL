# key_logger
# stack, implementation, greedy
# [left stack] cursor [right stack]


test_case = int(input())

for _ in range(test_case):
    left_stack = []
    right_stack = []
    data = input()
    for i in data:
        if i == '-':       # '-' means Backspace 
            if left_stack: # if there is no character on the left of the cursor, nothing can be erased
                left_stack.pop()
        elif i == '>':     # '>' means right arrow
            if right_stack:
                left_stack.append(right_stack.pop())
        elif i == '<':     # '<' means left arrow
            if left_stack:
                right_stack.append(left_stack.pop())
        else:
            left_stack.append(i)

    left_stack.extend(reversed(right_stack))  # Order of insertion in left stack : <-    / right stack : ->
    print(''.join(left_stack))