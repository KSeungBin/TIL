from typing import *
import collections

# priority queue
def leastInterval(self, tasks: List[str], n: int) -> int:
    counter = collections.Counter(tasks)
    result = 0

    while True:
        sub_count = 0
        # extracting counts
        for task, _ in counter.most_common(n + 1):
            sub_count += 1
            result += 1

            counter.subtract(task)
            # completely remove items below 0 from the list
            counter += collections.Counter()
        
        if not counter:
            break

        result += n - sub_count + 1
    
    return result