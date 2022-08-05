# method 1 : convert to list
import collections
from curses.ascii import isalnum
from typing import Counter, Deque


def isPalindrome(self, s:str) -> bool:
    strs = []
    
    for char in s:
        if char.isalnum():
            strs.append(char.lower())
    
    while len(strs) > 1:
        if strs.pop(0) != strs.pop:
            return False
    return True



# method 2 : optimization using deque 
def isPalindrome2(self, s:str) -> bool:
    strs: Deque = collections.deque()
    
    for char in s:
        if char.isalnum():
            strs.append(char.lower())
    
    while len(strs) > 1:
        if strs.popleft() != strs.pop():
            return False
    return True



# method 3 : optimization using slicing and regular expression
import re

def isPalindrome3(self, s:str) -> bool:
    s = s.lower()
    s = re.sub('[^a-z0-9]', '', s)

    return s == s[::-1]
    