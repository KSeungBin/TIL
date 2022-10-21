class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    longest: int = 0

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        def dfs(node: TreeNode) -> int:
            if not node:
                return -1
            # Search until reaching leaf nodes(both left and right child nodes of it are NULL) in a binary tree
            left = dfs(node.left)
            right = dfs(node.right)

            # longest path
            self.longest = max(self.longest, left + right + 2)
            # state
            return max(left, right) + 1
        
        dfs(root)
        return self.longest