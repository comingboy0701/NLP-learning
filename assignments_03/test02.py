# encoding: utf-8
"""
@author: chen km
@contact: 760855003@qq.com
@file: test02.py
@time: 2019/7/25 15:55
"""
# 动态规划
from collections import defaultdict
from collections import Counter
from functools import wraps
# 1 装饰器
# （1） 复杂的写法
call_time_with_arg = defaultdict(int)


def add_ten(n): return n + 10


def get_call_time(f):
    """@param f is a function"""
    @ wraps(f)  # 为了使 help(函数) 是显示 f 函数的帮助
    def wrap(n):
        result = f(n)
        call_time_with_arg[(f.__name__, n)] += 1
        return result
    return wrap


add_ten = get_call_time(add_ten)
add_ten(20)

# （2） 简单的写法
call_time_with_arg = defaultdict(int)
@get_call_time
def add_twenty(n): return n + 20


add_twenty(20)


original_price = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30, 35]
price = defaultdict(int)
for i, p in enumerate(original_price):
    price[i + 1] = p


call_time_with_arg = defaultdict(int)


def memo(f):
    memo.cache = {}

    def _wrap(arg):
        if arg in memo.cache:
            return memo.cache[arg]
        else:
            memo.cache[arg] = f(arg)
            # call_time_with_arg[(f.__name__, arg)] += 1
            return memo.cache[arg]
    return _wrap


solution = {}


@get_call_time
@memo
def r(n):
    max_price, max_split = max([(price[n], 0)] + [(r(i) + r(n - i), i)
                                                  for i in range(1, n)], key=lambda x: x[0])
    solution[n] = (n - max_split, max_split)
    return max_price


def parse_solution(n):
    left_split, right_split = solution[n]
    if right_split == 0:
        return [left_split]
    return parse_solution(left_split) + parse_solution(right_split)


Counter(call_time_with_arg).most_common()
r(50)
parse_solution(50)
