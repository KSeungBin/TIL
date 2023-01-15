from typing import *


# method 1: brute-force
def maxSlidingWindow1(self, nums: List[int], k: int) -> List[int]:
    if not nums:
        return nums
    
    r = []
    for i in range(len(nums) - k + 1):
        r.append(max(nums[i:i + k]))
    
    return r


# method 2: queuing optimization
import collections
def maxSlidingWindow2(self, nums: List[int], k: int) -> List[int]:
    results = []
    window = collections.deque()
    current_max = float('-inf')
    for i, v in enumerate(nums):
        window.append(v)
        if i < k - 1:
            continue

        if current_max == float('inf'):
            current_max = max(window)
        elif v > current_max:
            current_max = v
        
        results.append(current_max)

        if current_max == window.popleft():
            current_max = float('-inf')
    return results