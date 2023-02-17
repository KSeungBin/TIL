from typing import *


# method 1: brute-force
def majorityElement(self, nums: List[int]) -> int:
    for num in nums:
        if nums.count(num) > len(nums) // 2:
            return num