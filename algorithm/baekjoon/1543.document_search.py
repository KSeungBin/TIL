# document search
# greedy algorithm
# The split() method splits string into a list. 
# You can specify the separator, default separator is any whitespace.
# Note : When maxsplit is specified, the list will contain the specified nubner of elements plus one


document = input()
word = input()
count = 0
n = 0

while n <= len(document) - len(word):
    if document[n: n+len(word)] == word :
        count += 1
        n += len(word)
    else:
        n += 1
print(count)
    
    
    

