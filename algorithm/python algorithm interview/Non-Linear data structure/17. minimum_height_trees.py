# remove all leaf nodes from the undirected graph
# Among all possible rooted trees, those with minimum height are called minimum height trees(MHTs).
# Return a list of all MHTs' root labels. you can return the answer in any order
from typing import *
import collections

def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
    if n <= 1:
        return [0]
    
    # configuration of bidirectional graph
    graph = collections.defaultdict(list)
    for i, j in edges:
        graph[i].append(j)
        graph[j].append(i)
    
    # add the first leaf node
    leaves = []
    for i in range(n+1):
        if len(graph[i]) == 1:
            leaves.append(i)
    
    # iterated elimination until only root node remains
    while n > 2:
        n -= len(leaves)
        new_leaves = []
        for leaf in leaves:
            neighbor = graph[leaf].pop()
            graph[neighbor].remove(leaf)

            if len(graph[neighbor]) == 1:
                new_leaves.append(neighbor)
        
        leaves = new_leaves
    
    return leaves
    
