class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# method 1: brute-force search with recursive DFS
def rangeSumBST1(self, root: TreeNode, L: int, R: int) -> int:
    if not root:
        return 0
    
    return (root.val if L <= root.val <= R else 0) + \
            self.rangeSumBST(root.left, L, R) + \
            self.rangeSumBST(root.right, L, R)

# method 2: remove nodes that are not necessary using DFS with pruning
def rangeSumBST2(self, root: TreeNode, L: int, R: int) -> int:
    def dfs(node: TreeNode):
        if not node:
            return 0
        
        if node.val < L:
            return dfs(node.right)
        elif node.val > R:
            return dfs(node.left)
        return node.val + dfs(node.left) + dfs(node.right)
    
    return dfs(root)

# method 3: search necessary nodes using iterative DFS
def rangeSumBST3(self, root: TreeNode, L: int, R: int) -> int:
    stack, sum = [root], 0
    # implementation of iterative DFS using stack
    while stack:
        node = stack.pop()
        if node:
            if node.val > L:
                stack.append(node.left)
            if node.val < R:
                stack.append(node.right)
            if L <= node.val <= R:
                sum += node.val
    return sum

# method 4: search necessary nodes using iterative BFS
def rangeSumBST4(self, root: TreeNode, L: int, R: int) -> int:
    stack, sum = [root], 0
    # implementation of iterative BFS using basic operations of queue
    while stack:
        node = stack.pop(0)
        if node:
            if node.val > L:
                stack.append(node.left)
            if node.val < R:
                stack.append(node.right)
            if L <= node.val <= R:
                sum += node.val
    return sum