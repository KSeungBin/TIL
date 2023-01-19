# 424. longest repeating character replacement
import collections


# method: two-pointer, sliding window, Counter
def characterReplacement(self, s: str, k: int) -> int:
    left = right = 0
    counts = collections.Counter()
    for right in range(1, len(s) + 1):
        counts[s[right - 1]] += 1
        # search the most common characters
        max_char_n = counts.most_common(1)[0][1]

        # move left pointer when the result of the operation exceeds k
        if right - left - max_char_n > k:
            counts[s[left]] -= 1
            left += 1
    return right - left