import collections


def groupAnagrams(self, strs:list(str)) -> list(list(str)):
    anagrams = collections.defaultdict(list)

    for word in strs:
        anagrams[''.join(sorted(word))].append(word)
    return list(anagrams.values())


# sort() method in List data type : in-place-sort, override input values, no return

# sorted() method : 'key=' option 
a = ['cbe', 'cfc', 'abc']
sorted(a, key=lambda x: [x[0], x[-1]])  # 1st : x[0], 2nd : x[-1]
