# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Method 1: Insertion sort
def insertionSortList1(self, head: ListNode) -> ListNode:
    cur = parent = ListNode(None)
    while head:
        while cur.next and cur.next.val < head.val:
            cur = cur.next
        
        cur.next, head.next, head = head, cur.next, head.next

        cur = parent
    return cur.next

# Method 2: Improving comparison condition for insertion sort
def insertionSortList2(self, head: ListNode) -> ListNode:
    # change the initial value
    cur = parent = ListNode(0)
    while head:
        while cur.next and cur.next.val < head.val:
            cur = cur.next
        
        cur.next, head.next, head = head, cur.next, head.next
        
        # the 'cur' pointer goes back only when necessary 
        if head and cur.val > head.val:
            cur = parent
    return parent.next