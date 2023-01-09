# method 1: count the number of 1's
def hammingWeight(self, n: int) -> int:
    return bin(n).count('1')