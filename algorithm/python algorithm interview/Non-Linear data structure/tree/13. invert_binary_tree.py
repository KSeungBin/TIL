class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# method 1: pythonic way
def invertTree1(self, root:TreeNode) -> TreeNode:
    if root:
        root.left, root.right = \
            self.invertTree(root.right), self.invertTree(root.left)
        return root


# method 2: iterative implementation of BFS
import collections
def invertTree2(self, root:TreeNode) -> TreeNode:
    queue = collections.deque([root])

    while queue:
        node = queue.popleft()
        # top-down swap from parent node
        if node:
            node.left, node.right = node.right, node.left

            queue.append(node.left)
            queue.append(node.right)
    
    return root


# method 3: iterative implementation of DFS
def invertTree3(self, root:TreeNode) -> TreeNode:
    stack = collections.deque([root])

    while stack:
        node = stack.pop()
        # top-down swap from parent node
        if node:
            node.left, node.right = node.right, node.left

            stack.append(node.left)
            stack.append(node.right)
    
    return root


# method 4: iterative post-order traversal of binary tree using DFS
def invertTree4(self, root:TreeNode) -> TreeNode:
    stack = collections.deque([root])

    while stack:
        node = stack.pop()

        if node:
            stack.append(node.left)
            stack.append(node.right)

            node.left, node.right = node.right, node.left # post-order traversal
    
    return root