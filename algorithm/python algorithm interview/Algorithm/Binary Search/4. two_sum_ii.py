from typing import *
import bisect

# method 1: two-pointer
def twoSum1(self, numbers: List[int], target: int) -> List[int]:
    left, right = 0, len(numbers) - 1
    while not left == right:
        if numbers[left] + numbers[right] < target:
            left += 1
        elif numbers[left] + numbers[right] > target:
            right -= 1
        else:
            return left + 1, right + 1


# method 2: binary search
def twoSum2(self, numbers: List[int], target: int) -> List[int]:
    for k, v in enumerate(numbers):
        left, right = k + 1, len(numbers) - 1
        expected = target - v

        while left <= right:
            mid = left + (right - left) // 2
            if numbers[mid] < expected:
                left = mid + 1
            elif numbers[mid] > expected:
                right = mid - 1
            else:
                return k + 1, mid + 1


# method 3: bisect module + slicing
def twoSum3(self, numbers: List[int], target: int) -> List[int]:
    for k, v in enumerate(numbers):
        expected = target - v
        i = bisect.bisect_left(numbers[k + 1:], expected)
        if i < len(numbers[k + 1:]) and numbers[i + k + 1] == expected:
            return k + 1, i + k + 2


# method 4: bisect module + remove slicing
def twoSum4(self, numbers: List[int], target: int) -> List[int]:
    for k, v in enumerate(numbers):
        expected = target - v
        i = bisect.bisect_left(numbers, expected, k + 1)
        if i < len(numbers) and numbers[i] == expected:
            return k + 1, i + 1