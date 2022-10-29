from typing import *

# method 1: Using heapq module
import heapq
def findKthLargest1(self, nums: List[int], k: int) -> int:
    heap = list()
    for n in nums:
        heapq.heappush(heap, -n)
    
    for _ in range(1, k):
        heapq.heappop(heap)
    
    return -heapq.heappop(heap)


# method 2: Using heapify() method of heapq module
def findKthLargest2(self, nums: List[int], k: int) -> int:
    heapq.heapify(nums)

    for _ in range(len(nums) - k):
        heapq.heappop(nums)
    
    return heapq.heappop(nums)


# method 3: Using nlargest() method of heapq module
def findKthLargest3(self, nums: List[int], k: int) -> int:
    return heapq.nlargest(k, nums)[-1]


# method 4: sorting
def findKthLargest4(self, nums: List[int], k: int) -> int:
    return sorted(nums, reverse=True)[k-1]