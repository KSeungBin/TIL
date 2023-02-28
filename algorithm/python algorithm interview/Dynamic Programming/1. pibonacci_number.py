# method 1: recursive structure, brute-force
def fib(self, N: int) -> int:
    if N <= 1:
        return N
    return self.fib(N - 1) + self.fib(N - 2)


# method 2: memoization
import collections
class Solution:
    dp = collections.defaultdict(int)

    def fib(self, N: int) -> int:
        if N <= 1:
            return N
        
        if self.dp[N]:
            return self.dp[N]
        self.dp[N] = self.fib(N - 1) + self.fib(N - 2)
        return self.dp[N]