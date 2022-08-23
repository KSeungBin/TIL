# https://leetcode.com/problems/add-two-numbers/
from typing import *


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    # Generate reversed linked list
    def reverseList(self, head:ListNode) -> ListNode:
        node, prev = head, None
        while node:
            next, node.next = node.next, prev
            prev, node = node, next
        return prev
    
    # Convert linked list to Python list
    def toList(self, node: ListNode) -> List:
        list: List = []
        while node:
            list.append(node.val)
            node = node.next
        return list
    
    # Convert Python list to Linked list
    def toReversedLinkedList(self, result:str) -> ListNode:
        prev: ListNode = None
        for r in result:
            node = ListNode(r)
            node.next = prev
            prev = node
        return node
    
    # Add two linked lists
    def addTwoNumbers(self, l1:ListNode, l2:ListNode) -> ListNode:
        a = self.toList(self.reverseList(l1))
        b = self.toList(self.reverseList(l2))

        resultStr = int(''.join(str(e) for e in a)) + \
                    int(''.join(str(e) for e in b))
        
        return self.toReversedLinkedList(str(resultStr))