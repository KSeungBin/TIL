import collections


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class MyHashMap:
    def __init__(self):
        self.size = 1000
        self.table = collections.defaultdict(ListNode)
    
    # insertion
    def put(self, key:int, value:int) -> None:
        index = key % self.size
        # if there are no nodes in the index, insert and exit
        if self.table[index].value is None:
            self.table[index] = ListNode(key, value)
            return
        
        # if node exists in the index, generate linked list
        p = self.table[index]
        while p:
            if p.key == key:
                p.value = value
                return
            if p.next is None:
                break
            p = p.next
        p.next = ListNode(key, value)
    
    # search
    def get(self, key:int) -> int:
        index = key % self.size
        if self.table[index].value is None:
            return -1
        
        # if node exists, search matched key
        p = self.table[index]
        while p:
            if p.key == key:
                return p.value
            p = p.next
        return -1
    
    # deletion
    def remove(self, key:int) -> None:
        index = key % self.size
        if self.table[index].value is None:
            return
        
        # delete when it is the first node in the index
        p = self.table[index]
        if p.key == key:
            self.table[index] = ListNode() if p.next is None else p.next
            return
        
        # delete linked list node
        prev = p
        while p:
            if p.key == key:
                prev.next = p.next
                return
            prev, p = p, p.next