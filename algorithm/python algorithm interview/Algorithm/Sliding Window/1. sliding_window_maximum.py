from typing import *


# method 1: brute-force
def maxSlidingWindow1(self, nums: List[int], k: int) -> List[int]:
    if not nums:
        return nums
    
    r = []
    for i in range(len(nums) - k + 1):
        r.append(max(nums[i:i + k]))
    
    return r

