# method 1 : brute-force (time out)
def maxProfit1(self, prices:list(int)) -> int:
    max_price = 0
    for i, price in enumerate(prices):
        for j in range(i, len(prices)):
            max_price = max(prices[j]-price, max_price)

    return max_price


# method 2 : Kadane's Algorithm
import sys
def maxProfit2(self, prices:list(int)) -> int:
    profit = 0   # profit = -sys.maxsize = float('-inf')
    min_price = sys.maxsize  # float('inf')

    for price in prices:
        min_price = min(price, min_price)
        profit = max(profit, price - min_price)
        
    return profit
