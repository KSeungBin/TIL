# method 1 : Brute-Force
def twoSum1(nums:list(int), target:int) -> list(int):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i,j]


# method 2 : Using the 'in' operator to search for item in a list
def twoSums2(nums:list(str), target:int) -> list(int):
    for i, num in enumerate(nums):
        complement = target - num

        if complement in nums[i+1:]:
            return [nums.index[num], nums[i+1:].index[complement] + (i+1)]


# method 3 : Generate dictionary and look up the key
def twoSums3(nums:list(str), target:int) -> list(int):
    nums_map = {}
    # switch key and value
    for i, num in enumerate(nums):
        nums_map[num] = i
    
    # subtract the first number and look up the key
    for i, num in enumerate(nums):
        if target - num in nums_map and i != nums_map[target - num]:
            return [i, nums_map[target - num]]


# method 4 : Improving lookup architecture
def twoSums4(nums:list(str), target:int) -> list(int):
    nums_map = {}
    # merge the code from 2 for-loops into 1 for-loop
    for i, num in enumerate(nums):
        if target - num in nums_map:
            return [nums_map[target-num], i]
        nums_map[num] = i


# Two-pointer can't be used in this problem.
# beacause index value has to be maintained to solve this problem
def twoSum5(nums:list(int), target:int) -> list(int):
    nums.sort()  # sorting isn't allowed in this case
    left, right = 0, len(nums) - 1
    while not left == right:
        if nums[left] + nums[right] < target:
            left += 1
        elif nums[left] + nums[right] > target:
            right -= 1
        else:
            return [left, right]






