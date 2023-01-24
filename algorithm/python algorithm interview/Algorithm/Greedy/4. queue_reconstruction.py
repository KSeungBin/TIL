# Queue reconstruction by height
from typing import *
import heapq

# priority queue
def reconstructionQueue(self, people: List[List[int]]) -> List[List[int]]:
    heap = []
    # height in reverse order, insert index
    for person in people:
        heapq.heappush(heap, (-person[0], person[1]))
    
    result = []
    # height in reverse order, extract index
    while heap:
        person = heapq.heappop(heap)
        result.insert(person[1], [-person[0], person[1]])
    return result