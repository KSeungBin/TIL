from typing import *

# method 1: visit all
def canCompleteCircuit1(self, gas: List[int], cost: List[int]) -> int:
    for start in range(len(gas)):
        fuel = 0
        for i in range(start, len(gas) + start):
            index = i % len(gas)

            can_travel = True
            if gas[index] + fuel < cost[index]:
                can_travel = False
                break
            else:
                fuel += gas[index] - cost[index]
        if can_travel:
            return start
    return -1


# method 2: visit once
def canCompleteCircuit2(self, gas: List[int], cost: List[int]) -> int:
    if sum(gas) < sum(cost):
        return -1
    
    start, fuel = 0, 0
    for i in range(len(gas)):
        if gas[i] + fuel < cost[i]:
            start = i + 1
            fuel = 0
        else:
            fuel += gas[i] - cost[i]
    return start