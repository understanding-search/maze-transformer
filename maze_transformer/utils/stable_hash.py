import hashlib
import struct

def stable_hash(s: str) -> int:
    """Returns a stable hash of the given string. not cryptographically secure, but stable between runs"""
    # init hash object and update with string
    hash_obj: hashlib._Hash = hashlib.sha256()
    hash_obj.update(bytes(s, "UTF-8"))
    # get digest and convert to int
    return int.from_bytes(hash_obj.digest(), 'big')
