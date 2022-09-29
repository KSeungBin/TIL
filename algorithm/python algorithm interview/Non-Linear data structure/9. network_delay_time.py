import collections
import heapq
from typing import *
def networkDelayTime(self, times: List[List[int]], N: int, K: int) -> int:
    graph = collections.defaultdict(list)
    # adjacency list representation of graph
    for u, v, w in times:
        graph[u].append((v,w))

    # variable Q: [(taken time, vertex)]
    Q = [(0, K)]
    dist = collections.defaultdict(int)

    # insert shortest path to vertex with min-priority queue
    while Q:
        time, node = heapq.heappop(Q)
        if node not in dist:
            dist[node] = time
            for v,w in graph[node]:
                alt = time + w
                heapq.heappush(Q, (alt, v))
    
    # detect whether the shortest path exists for all nodes
    if len(dist) == N:
        return max(dist.values())
    return -1
         