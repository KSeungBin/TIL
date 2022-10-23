from turtle import right


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    result: int = 0

    def longestUnivaluePath(self, root:TreeNode) -> int:
        def dfs(node: TreeNode):
            if node in None:
                return 0
            
            # recursive dfs to non-existent node
            left = dfs(node.left)
            right = dfs(node.right)

            # increase distance by 1 if current node is equal to child node
            if node.left and node.left.val == node.val:
                left += 1
            else:
                left = 0
            if node.right and node.right.val == node.val:
                right += 1
            else:
                right = 0

            # the result is the maximum sum of the distances between the left and right child nodes
            self.result = max(self.result, left + right)
            # return the largest value of the child node status
            return max(left, right)

        dfs(root)
        return self.result