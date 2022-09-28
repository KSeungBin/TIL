import collections
from typing import *
# method 1 : detect a cycle using dfs
def canFinish1(self, numCourses:int, prerequisites:List[List[int]]) -> bool:
    graph = collections.defaultdict(list)

    for x, y in prerequisites:
        graph[x].append(y)
    
    traced = set()

    def dfs(i):
        # If the current node exists in traced, a set of nodes that have already been visited, it is a cyclic structure.
        if i in traced:
            return False

        traced.add(i)
        for y in graph[i]:
            if not dfs(y):
                return False
        
        # remove cyclic node after search
        traced.remove(i)

        return True
    
    # detect cyclic structure
    for x in list(graph):
        if not dfs(x):
            return False
    
    return True


# method 2 : optimization using pruning
def canFinish2(self, numCourses:int, prerequisites:List[List[int]]) -> bool:
    graph = collections.defaultdict(list)
    for x, y in prerequisites:
        graph[x].append(y)
    
    traced = set()
    visited = set()

    def dfs(i):
        # If it is a cyclic structure, return False
        if i in traced:
            return False
        # Return True if the node has already been visited
        if i in visited:
            return True
        
        traced.add(i)
        for y in graph[i]:
            if not dfs(y):
                return False
        # delete circular node after search
        traced.remove(i)
        # add visited node after search
        visited.add(i)

        return True
    
    # detect cyclic structure
    for x in list(graph):
        if not dfs(x):
            return False
    
    return True