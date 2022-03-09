from itertools import islice
from typing import Iterable, Tuple


def window(seq: Iterable, n=2) -> Tuple:
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
