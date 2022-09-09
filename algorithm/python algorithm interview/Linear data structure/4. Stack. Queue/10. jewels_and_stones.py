# method 1 : hash table
def numJuwelsInStones1(self, J:str, S:str) -> int:
    freqs = {}
    count = 0

    # count frequency of stones(S)
    for char in S:
        if char not in freqs:
            freqs[char] = 1
        else:
            freqs[char] += 1
    
    # count frequency of jewels(J)
    for char in J:
        if char in freqs:
            count += freqs[char]
    return count


# method 2 : skip comparison using defaultdict
import collections


def numJuwelsInStones2(self, J:str, S:str) -> int:
    freqs = collections.defaultdict(int)
    count = 0

    # counting stones(S) frequency without comparison
    for char in S:
        freqs[char] += 1
    
    # sum the number of jewels(J) frequency without comparison
    for char in J:
        count += freqs[char]
    return count


# method 3 : skip calculation using Counter
def numJuwelsInStones3(self, J:str, S:str) -> int:
    freqs = collections.Counter(S) # count stones(S) frequency
    count = 0

    # sum the number of jewels(J) frequency without comparison
    for char in J:
        count += freqs[char]
    return count


# method 4 : pythonic way
def numJuwelsInStones4(self, J:str, S:str) -> int:
    return sum(s in J for s in S)