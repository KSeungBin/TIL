import itertools
from typing import *


# method 1 : dfs
def combine(self, n:int, k:int) -> List[List[int]]:
    results = []

    def dfs(elements, start:int, k:int):
        if k == 0:
            results.append(elements[:])
            return
        
        for i in range(start, n+1):
            elements.append(i)
            dfs(elements, i+1, k-1)
            elements.pop()
    
    dfs([], 1, k)
    return results


# method 2 : itertools module
def combine(self, n:int, k:int) -> List[List[int]]:
    return list(itertools.combinations(range(1, n+1), k))
    