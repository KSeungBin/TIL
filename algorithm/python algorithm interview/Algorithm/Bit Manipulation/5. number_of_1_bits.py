# method 1: count the number of 1's
def hammingWeight(self, n: int) -> int:
    return bin(n).count('1')


# method 2: bitwise operation
def hammingWeight2(self, n: int) -> int:
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count