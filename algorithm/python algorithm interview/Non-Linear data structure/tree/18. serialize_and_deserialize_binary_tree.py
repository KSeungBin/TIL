# Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, 
# or transmitted across a network connection link to be reconstructed later in the same or another computer environment.
# You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

import collections

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Codec:
    # Serialization
    def serialize(self, root: TreeNode) -> str:
        queue = collections.deque([root])
        result = ['#']
        # Serialize a tree using BFS
        while queue:
            node = queue.popleft()
            if node:
                queue.append(node.left)
                queue.append(node.right)

                result.append(str(node.val))
            else:
                result.append('#')
        return ' '.join(result)
    
    # Deserialization
    def deserialize(self, data: str) -> TreeNode:
        # Exception handling
        if data == '# #':
            return None
        
        nodes = data.split()

        root = TreeNode(int(nodes[1]))
        queue = collections.deque([root])
        index = 2
        # Like a fast runner, check the child nodes results first and then insert the queue
        while queue:
            node = queue.popleft()
            if nodes[index] is not '#':
                node.left = TreeNode(int(nodes[index]))
                queue.append(node.left)
            index += 1

            if nodes[index] is not '#':
                node.right = TreeNode(int(nodes[index]))
                queue.append(node.right)
            index += 1
        return root