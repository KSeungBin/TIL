# calculate the difference in height using recursive structure
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isBalanced(self, root: TreeNode) -> bool:
    def check(root):
        if not root:
            return 0
        
        left = check(root.left)
        right = check(root.right)
        
        # -1 if there is a difference in height, otherwise increase by 1 depending on the height
        if left == -1 or \
            right == -1 or \
                abs(left - right) > 1:
                return -1
        return max(left, right) + 1
    
    return check(root) != -1