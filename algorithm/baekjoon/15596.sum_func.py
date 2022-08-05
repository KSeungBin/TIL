
# a : Array containing n integers to be summed
# return value : sum of n integers in a


# method 1
def solve1(a:list) -> int:
  ans = 0
  for number in a:
    ans += number
  return ans


# method 2
def solve2(a:list) -> int:
    return sum(a)
    