# method 1 : recursive
def reverseList(self, head):
    def reverse(node, prev=None):
        if not node:
            return prev
        next, node.next = node.next, prev
        return reverse(next, node)
    return reverse(head)


# mehod 2 : iterative (lower space complexity)
def reverseList(self, head):
    node, prev = head, None
    while node:
        next, node.next = node.next, prev
        prev, node = node, next
    return prev

