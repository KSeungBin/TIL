# method 1 : Convert to list
def isPalindrome1(self, head):
    q = []

    if not head:
        return True
    
    node = head
    while node is not None:
        q.append(node.val)
        node = node.next
    
    while len(q) > 1:
        if q.pop(0) != q.pop():
            return False
    return True


# method 2 : Optimization using deque
import collections
def isPalindrome2(self, head):
    q = collections.deque()

    if not head:
        return True
    
    node = head
    while node is not None:
        q.append(node.val)
        node = node.next
    
    while len(q) > 1:
        if q.popleft() != q.pop():
            return False
    return True


# method 3 : Runner + Multi Assignment
def isPalindrome3(self, head):
    rev = None
    slow = fast = head

    while fast and fast.next:
        fast = fast.next.next
        rev, rev.next, slow = slow, rev, slow.next
    if fast:
        slow = slow.next

    while rev and rev.val == slow.val:
        slow, rev = slow.next, rev.next
    return not rev 
