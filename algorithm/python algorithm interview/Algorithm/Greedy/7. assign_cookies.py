from typing import *
import bisect

# method 1: greedy algorithm
def findContentChildren1(self, g: List[int], s: List[int]) -> int:
    g.sort()
    s.sort()

    child_i = cookie_j = 0
    # greedy
    while child_i < len(g) and cookie_j < len(s):
        if s[cookie_j] >= g[child_i]:
            child_i += 1
        cookie_j += 1
    
    return child_i


# method 2: binary search
def findContentChildren2(self, g: List[int], s: List[int]) -> int:
    g.sort()
    s.sort()

    result = 0
    for i in s:
        # search larger index
        index = bisect.bisect_right(g, i)
        if index > result:
            result += 1
    return result