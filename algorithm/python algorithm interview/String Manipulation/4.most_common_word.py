# method 1 : Using defaultdict
import collections
import re

def mostCommonWord1(self, paragraph:str, banned:list(str)) -> str:
    # Data cleansing : data preprocessing on input
    words = [word for word in re.sub(r'[^\w]', ' ', paragraph).lower().split()
                if not word in banned]
    
    counts = collections.defaultdict(int)
    for word in words:
        counts[word] += 1
    return max(counts, key=counts.get)

     

# method 2 : Using list comprehension and Counter
def mostCommonWord2(self, paragraph:str, banned:list(str)) -> str:
    words = [word for word in re.sub(r'[^\w]',' '. paragraph).lower().split()
                if word not in banned]
    
    counts = collections.Counter(words)
    return counts.most_common(1)[0][0]

