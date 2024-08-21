
import random

def multi_random_int_in_range(num, start, end, include_end=False, allow_repeat=False):
    # assert isinstance(num, int)
    if include_end:
        end += 1
    if allow_repeat:
        return random.choices(range(start, end), k=num)
    else:
        return multi_random_int_in_range_no_repeat(num, start, end, include_end)

def multi_random_int_in_range_no_repeat(num, start, end, include_end=False):
    if include_end:
        end += 1
    assert end > start
    assert end - start >= num
    return random.sample(range(start, end), num)


from .sample import (
    sample_from_gaussian_01
)