# Given an integer array nums where the elements are sorted in ascending order, convert it to a height-balanced binary search tree.
# A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one.

from typing import *

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
    if not nums:
        return None
    
    mid = len(nums) // 2

    # Construct a binary search tree using divide-and-conquer algorithm
    node = TreeNode(nums[mid])
    node.left = self.sortedArrayToBST(nums[:mid])
    node.right = self.sortedArrayToBST(nums[mid+1:])

    return node