# Given the root of a Binary Search Tree (BST), return the minimum difference between the values of any two different nodes in the tree.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# method 1: inorder traversal with recursion
import sys

class Solution:
    prev = -sys.maxsize
    result = sys.maxsize
    
    # comparison result of inorder traversal with recursion
    def minDiffInBST(self, root: TreeNode) -> int:
        if root.left:
            self.minDiffInBST(root.left)
        
        self.result = min(self.result, root.val - self.prev)
        self.prev = root.val

        if root.right:
            self.minDiffInBST(root.right)
        
        return self.result