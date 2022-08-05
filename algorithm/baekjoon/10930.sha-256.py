# hash, implementation
# hashlib module implements a common interface to many different secure hash and message digest algorithms
# The terms "secure hash" and "message digest" are interchangeable. The modern term is secure hash.


import hashlib

input_data = input()
encoded_data = input_data.encode()
result = hashlib.sha256(encoded_data).hexdigest()
print(result)