# https://leetcode.com/problems/swap-nodes-in-pairs/

# Definition for singly-linked list.
from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def swapPairs1(self, head: ListNode) -> ListNode:
        cur = head
        while cur and cur.next:
            cur.val, cur.next.val = cur.next.val, cur.val
            cur = cur.next.next
        return head
    
    # method 1 : iterative
    def swapPairs2(self, head:ListNode) -> ListNode:
        root = prev = ListNode(None)
        prev.next = head
        while head and head.next:
            # b -> a
            b = head.next
            head.next = b.next
            b.next = head

            # prev -> b
            prev.next = b

            # Go to next comparison
            head = head.next
            prev = prev.next.next
        return root.next

    # method 2 : recursive (lower space complexity)
    def swapPairs3(self, head:ListNode) -> ListNode:
        if head and head.next:
            p = head.next
            head.next = self.swapPairs3(p.next)
            p.next = head
            return p
        return head
    
