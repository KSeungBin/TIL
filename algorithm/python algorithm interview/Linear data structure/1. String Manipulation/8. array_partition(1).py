# mehod 1 : ascending pair
def arrayPairSum1(self, nums:list(int)) -> int:
    sum = 0
    pair = []
    nums.sort()

    for n in nums:
        pair.append(n)
        if len(pair) == 2:
            sum += min(pair)
            pair = []
    return sum


# method 2 : calculate numbers at even index
def arrayPairSum2(self, nums:list(int)) -> int:
    sum = 0
    nums.sort()

    for i, n in enumerate(nums):
        if i % 2 == 0:
            sum += n 
    return sum


# method 3 : pythonic way (slicing)
def arrayPairSum3(self, nums:list(int)) -> int:
    return sum(sorted(nums)[::2])





