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