import collections
import heapq
# method 1 : extracting net negative numbers using Counter
from typing import *
def topKFrequent(self, nums:List[int], k:int) -> List[int]:
    freqs = collections.Counter(nums)
    freqs_heap = []

    # Insert negative numbers into the heap
    for f in freqs:
        heapq.heappush(freqs_heap, (-freqs[f], f))
    
    topk = list()
    # extract k times
    # since it is the min heap, extracting in the order of the smallest negative number
    for _ in range(k):
        topk.append(heapq.heappop(freqs_heap)[1])

    return topk


# method 2 : pythonic way
def topKFrequent(self, nums, k):
    return list(zip(*collections.Counter(nums).most_common(k)))[0]
