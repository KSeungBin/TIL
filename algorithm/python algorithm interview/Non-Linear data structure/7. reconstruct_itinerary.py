import collections
from typing import *


# method 1 : dfs
def findItinerary1(self, tickets:List[List[str]]) -> List[str]:
    graph = collections.defaultdict(list)
    for a, b in sorted(tickets):
        graph[a].append(b)
    
    route = []
    def dfs(a):
        while graph[a]:
            dfs(graph[a].pop(0))
        route.append(a)
    
    dfs('JFK')
    return route[::-1]


# method 2 : stack
def findItinerary2(self, tickets:List[List[str]]) -> List[str]:
    graph = collections.defaultdict(list)
    for a, b in sorted(tickets, reverse=True):
        graph[a].append(b)
    
    route, stack = [], ['JFK']
    while stack:
        while graph[stack[-1]]:
            stack.append(graph[stack[-1]].pop(0))
        route.append(stack.pop())
    
    return route[::-1]