# method 1: search at the end of first row
def searchMatrix1(self, matrix, target):
    # exception processing
    if not matrix:
        return False
    
    # end of first row
    row = 0
    col = len(matrix[0]) - 1

    while row <= len(matrix) - 1 and col >= 0:
        if target == matrix[row][col]:
            return True
        # move left if target is small
        elif target < matrix[row][col]:
            col -= 1
        # move downward if target is large
        elif target > matrix[row][col]:
            row += 1
    return False


# method 2: pythonic way
def searchMatrix2(self, matrix, target):
    return any(target in row for row in matrix)