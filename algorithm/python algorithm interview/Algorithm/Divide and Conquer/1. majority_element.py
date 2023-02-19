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


# method 3: divide and conquer
def majorityElement3(self, nums: List[int]) -> int:
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    
    half = len(nums) // 2
    a = self.majorityElement3(nums[:half])
    b = self.majorityElement3(nums[half:])

    return [b,a][nums.count(a) > half]