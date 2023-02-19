from typing import *
import collections

# method 1: brute-force
def majorityElement1(self, nums: List[int]) -> int:
    for num in nums:
        if nums.count(num) > len(nums) // 2:
            return num


# method 2: dynamic programming
def majorityElement2(self, nums: List[int]) -> int:
    counts = collections.defaultdict(int)
    for num in nums:
        if counts[num] == 0:
            counts[num] = nums.count(num)
        
        if counts[num] > len(nums) // 2:
            return num