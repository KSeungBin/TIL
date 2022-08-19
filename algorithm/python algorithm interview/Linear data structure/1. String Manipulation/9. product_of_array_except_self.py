from os import O_TEMPORARY


def productExceptSelf(self, nums:list(int)) -> list(int):
    out = []
    p = 1
    # Multiplication from the left
    for i in range(0, len(nums)):
        out.append(p)
        p = p * nums[i]
    p = 1
    # Multiply the result of multiplication from the left by the values from the right
    for i in range(len(nums)-1, 0-1, -1):
        out[i] = out[i] * p
        p = p * nums[i]
    return out

