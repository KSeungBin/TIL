from typing import *


# method 1: greedy algorithm
def maxProfit1(self, prices: List[int]) -> int:
    result = 0

    for i in range(len(prices) - 1):
        if prices[i + 1] > prices[i]:
            result += prices[i + 1] - prices[i]
    return result


# method 2: Pythonic way
def maxProfit2(self, prices: List[int]) -> int:
    return sum(max(prices[i + 1] - prices[i], 0) for i in range(len(prices) - 1))
    