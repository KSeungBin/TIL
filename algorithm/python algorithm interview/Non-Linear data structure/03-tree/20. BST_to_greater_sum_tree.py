# Given the root of a Binary Search Tree (BST), convert it to a Greater Tree 
# such that every key of the original BST is changed to the original key plus the sum of all keys greater than the original key in BST.

# As a reminder, a binary search tree is a tree that satisfies these constraints:
# 1. The left subtree of a node contains only nodes with keys less than the node's key.
# 2. The right subtree of a node contains only nodes with keys greater than the node's key.
# 3. Both the left and right subtrees must also be binary search trees.

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    val: int = 0

    def bstToGst(self, root: TreeNode) -> TreeNode:
        # Accumulate nodes' values using inorder traversal
        if root:
            self.bstToGst(root.right)
            self.val += root.val
            root.val = self.val
            self.bstToGst(root.left)
        
        return root