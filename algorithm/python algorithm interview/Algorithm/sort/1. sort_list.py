# Given the head of a linked list, return the list after sorting it in ascending order.

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
 
# method 1: merge sort
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    if l1 and l2:
        if l1.val > l2.val:
            l1, l2 = l2, l1
        l1.next = self.mergeTwoLists(l1.next, l2)
    
    return l1 or l2

def sortList(self, head: ListNode) -> ListNode:
    if not (head and head.next):
        return head
    
    # Runner Technique
    half, slow, fast = None, head, head
    while fast and fast.next:
        half, slow, fast = slow, slow.next, fast.next.next
    half.next = None

    # split recursion
    l1 = self.sortList(head)
    l2 = self.sortList(slow)

    return self.mergeTwoLists(l1, l2)


# method 2: practical way to use built-in functions in python
from typing import *
def sortList(self, head: ListNode) -> ListNode:
    # linked list -> python list
    p = head
    lst: List = []
    while p:
        lst.append(p.val)
        p= p.next
    
    # sort
    lst.sort()

    # python list -> linked list
    p = head
    for i in range(len(lst)):
        p.val = lst[i]
        p = p.next
    return head
    