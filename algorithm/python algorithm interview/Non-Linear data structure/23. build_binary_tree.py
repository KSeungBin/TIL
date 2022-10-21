# Construct binary tree from preorder and inorder traversal
# inorder traversal divide-and-conquer with the result of preorder traversal
from typing import *

# Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, 
# construct and return the binary tree.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
    if inorder:
        # result of preorder traversal = partitioned index of inorder traversal
        # Partition the inorder array into Left and Right sub-trees by getting the index  
        index = inorder.index(preorder.pop(0))

        # divide-and-conquer: results of inorder traversal
        node = TreeNode(inorder[index])
        node.left = self.buildTree(preorder, inorder[0:index])
        node.right = self.buildTree(preorder, inorder[index+1:])    

        return node