from typing import *

# method 1: recursion
def search1(self, nums: List[int], target: int) -> int:
    def binary_search(left, right):
        if left <= right:
            mid = (left + right) // 2

            if nums[mid] < target:
                return binary_search(mid + 1, right)
            elif nums[mid] > target:
                return binary_search(left, mid - 1)
            else:
                return mid
        else:
            return -1
    
    return binary_search(0, len(nums) - 1)

# method 2: iteration
def search2(self, nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2

        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            return mid
    return -1


# method 3: binary search module
import bisect
def search3(self, nums: List[int], target: int) -> int:
    index = bisect.bisect_left(nums, target)

    if index < len(nums) and nums[index] == target:
        return index
    else:
        return -1


# method 4: Applying index not using the technique of binary search
def search4(self, nums: List[int], target: int) -> int:
    try:
        return nums.index(target)
    except ValueError:
        return -1
