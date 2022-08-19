# method 1. Swap using two pointer
def reverseString1(self, s: list[str]) -> None:
    left, right = 0, len(s)-1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1


# method 2. pythonic way
# python List reverse() is an inbuilt method in the python programming language
def reverseString2(self, s:list[str]) -> None:
    s.reverse()

# Slicing a list, string, tuple in python
# s = s[::-1]
# O(1) : s[:] = s[::-1] 



