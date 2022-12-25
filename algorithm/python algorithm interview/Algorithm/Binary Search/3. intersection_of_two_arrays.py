from typing import *


# method 1: brute-force
def intersection1(self, nums1: List[int], nums2: List[int]) -> List[int]:
    result: Set = set()
    for n1 in nums1:
        for n2 in nums2:
            if n1 == n2:
                result.add(n1)
    
    return result


# method 2: binary search
import bisect
def intersection2(self, nums1: List[int], nums2: List[int]) -> List[int]:
    result: Set = set()
    nums2.sort()
    for n1 in nums1:
        i2 = bisect.bisect_left(nums2, n1)
        if len(nums2) > 0 and len(nums2) > i2 and n1 == nums2[i2]:
            result.add(n1)
    
    return result