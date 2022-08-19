# Using lambda and '+' operator
# a = [1,2,3], b = [4,5,6]
# b + a = [4,5,6,1,2,3]


def reorderLogFiles(self, logs:list[str]) -> list[str]:
    letters, digits = [], []
    for log in logs:
        if log.split(" ")[1].isdigit():
            digits.append(log)
        else:
            letters.append(log)
    
    letters.sort(key=lambda x: (x.split(" ")[1:], x.split(" ")[0]))
    letters + digits

# list comprehension is used to create list,
# Lambda is function that can process like other functions and thus return values or lists
# list comprehension is much more concise and readable.