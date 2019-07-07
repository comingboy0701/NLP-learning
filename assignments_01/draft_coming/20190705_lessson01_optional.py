# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:22:59 2019

@author: comingboy
"""


def is_variable(pat):
    return pat.startswith('?') and all(s.isalpha() for s in pat[1:]) # 字符串是否都是由字符组成

def pat_match(pattern, saying):
    if not pattern or not saying: return []

    if is_variable(pattern[0]):
        return [(pattern[0], saying[0])] + pat_match(pattern[1:], saying[1:])
    else:
        if pattern[0] != saying[0]:
            return []
        else:
            return pat_match(pattern[1:], saying[1:])

def pat_to_dict(patterns):
    return {k: ' '.join(v) if isinstance(v,list) else v for k, v in patterns}


def subsitite(rule, parsed_rules): # 字典的特性，如果键值存在value, 用get会得到value的值，否则返回 key值
    if not rule: return []

    return [parsed_rules.get(rule[0], rule[0])] + subsitite(rule[1:], parsed_rules)

pattern = '?P needs ?X'.split()
saying = "John needs vacation".split()
got_patterns = pat_match(pattern, saying)

dict_patterns = pat_to_dict(got_patterns)

subsitite_pat = subsitite("What if you mean if you got a ?X".split(), pat_to_dict(got_patterns))

join_pat = ' '.join(subsitite_pat)
print(join_pat)


def is_pattern_segment(pattern):
    return pattern.startswith('?*') and all(a.isalpha() for a in pattern[2:])

is_pattern_segment('?*9')

fail = [True, None]


def pat_match_with_seg(pattern, saying):
    if not pattern or not saying: return []

    pat = pattern[0]

    if is_variable(pat):
        return [(pat, saying[0])] + pat_match_with_seg(pattern[1:], saying[1:])
    elif is_pattern_segment(pat):
        match, index = segment_match(pattern, saying)
        return [match] + pat_match_with_seg(pattern[1:], saying[index:])
    elif pat == saying[0]:
        return pat_match_with_seg(pattern[1:], saying[1:])
    else:
        return fail


def segment_match(pattern, saying):
    seg_pat, rest = pattern[0], pattern[1:]
    seg_pat = seg_pat.replace('?*', '?')

    if not rest: return (seg_pat, saying), len(saying)

    for i, token in enumerate(saying):
        if rest[0] == token and is_match(rest[1:], saying[(i + 1):]):
            return (seg_pat, saying[:i]), i

    return (seg_pat, saying), len(saying)


def is_match(rest, saying):
    if not rest and not saying:
        return True
    if not all(a.isalpha() for a in rest[0]):
        return True
    if rest[0] != saying[0]:
        return False
    return is_match(rest[1:], saying[1:])


dict_patterns = pat_match_with_seg('?*P is very good and ?*X'.split(), "My dog is very good and my cat is very cute".split())




join_pat = ' '.join(subsitite('?P is very good'.split(), pat_to_dict(dict_patterns)))

print(join_pat)



def get_response(saying, rules):
    a, b = rules.keys(), rules.values()
    dict_patterns = pat_match_with_seg(list(a),saying.split())

    join_pat = ' '.join(subsitite(list(b)[0], pat_to_dict(dict_patterns)))
    return str(join_pat)

rules = {
    "?*X hello ?*Y": ["Hi, how do you do?"]}

saying = "I am mike, hello "

reponse = get_response(saying, rules)
print(reponse)