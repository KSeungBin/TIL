# method 1 : brute-force (time out)
def threeSum1(self, nums:list(int)) -> list(list(int)):
    result = []
    nums.sort()

    for i in range(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        for j in range(i+1, len(nums)-1):
            if j > i+1 and nums[j] == nums[j-1]:
                continue
            for k in range(j+1, len(nums)):
                if k > j+1 and nums[k] == nums[k-1]:
                    continue
                if nums[i] + nums[j] + nums[k] == 0:
                    result.append([nums[i],nums[j],nums[k]])
    return result


# method 2 : two-pointer (two pointers is an effective technique which is typically used for searching pairs in a "sorted" array.)
def threeSum2(self, nums:list(int)) -> list(list(int)):
    result = []
    nums.sort()

    for i in range(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue

        left, right = i+1, len(nums) - 1
        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if sum < 0:
                left += 1
            elif sum > 0:
                right -= 1
            else: # sum = 0
                result.append([nums[i], nums[left], nums[right]])
                # skip
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
    
    return result

                

