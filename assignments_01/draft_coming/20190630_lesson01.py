# encoding: utf-8
""" 
@author: chen km 
@contact: 760855003@qq.com 
@file: 20190630_lesson01.py 
@time: 2019/6/30 16:35 
"""
import random

def adj():
    return random.choice('蓝色的|好看的|小小的'.split('|'))

def adj_star():
    return random.choice([lambda :'', lambda :adj()+adj_star()])()

adj_star()


def create_grammar(grammar_str, split='=>', line_split='\n'):
    grammar = {}
    for line in grammar_str.split(line_split):
        if not line.strip(): continue
        exp, stmt = line.split(split)
        grammar[exp.strip()] = [s.split() for s in stmt.split('|')]
    return grammar

def generate(gram, target):
    if target not in gram: return target  # means target is a terminal expression

    expaned = [generate(gram, t) for t in random.choice(gram[target])]
    return ''.join([e if e !='/n' else '\n'for e in expaned if e != 'null'])
host = """
host => 寒暄 报数 询问 业务相关 结尾 
报数 => 我是 数字 号 ,
数字 => 单个数字 | 数字 单个数字 
单个数字 => 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
寒暄 => 称谓 打招呼 | 打招呼
称谓 => 人称 ,
人称 => 先生 | 女士 | 小朋友
打招呼 => 你好 | 您好 
询问 => 请问你要 | 您需要
业务相关 => 玩玩 具体业务
玩玩 => null
具体业务 => 喝酒 | 打牌 | 打猎 | 赌博
结尾 => 吗？
"""

simple_grammar = """
sentence => noun_phrase verb_phrase
noun_phrase => Article Adj* noun
Adj* => null | Adj Adj*
verb_phrase => verb noun_phrase
Article =>  一个 | 这个
noun =>   女人 |  篮球 | 桌子 | 小猫
verb => 看着   |  坐在 |  听着 | 看见
Adj =>  蓝色的 | 好看的 | 小小的
"""

programming = """
stmt => if_exp | while_exp | assignment 
assignment => var = var
if_exp => if ( var ) { /n .... stmt }
while_exp=> while ( var ) { /n .... stmt }
var => chars number
chars => char | char char
char => student | name | info  | database | course
number => 1 | 2 | 3
"""

print (generate(gram=create_grammar(simple_grammar), target='sentence'))

print(generate(gram=create_grammar(programming, split='=>'), target='stmt'))

print(generate(gram=create_grammar(host, split='=>'), target='host'))



import pandas as pd
import jieba
import re
from functools import reduce
from operator import add
from collections import Counter
import matplotlib.pyplot as plt
def token(string):
    return re.findall('\w+',string)

def cut(string):
    return list(jieba.cut(string))

filename = r'F:\eclipsewokspace\MachineLearing\NLP\datasource\sqlResult_1558435.csv'
content = pd.read_csv(filename,encoding='gb18030')
articless = content['content'].tolist()
articles = articless[:-1]
articles_clean = [ ''.join( token(str(a))) for a in articles]

# 这样写我电脑带不动，可能需要很大的内存吧
# words_count = Counter(reduce(add, [cut(a) for a in articles_clean]))

articles_cuts = []
for string in  articles_clean:
    articles_cuts +=cut(string)

words_count = Counter(articles_cuts)
words_count.most_common(100)
frequiences = [f for w, f in words_count.most_common(100)]
x = range(0,100)
plt.plot(x,frequiences)
plt.show()


def prob_2(word1, word2):
    if word1 + word2 in articles_cuts_gram2: return words_count_2[word1+word2] / words_count_len
    else:
        return 1 / len(articles_cuts_gram2)

articles_cuts_gram2 = [''.join(articles_cuts[i:i+2]) for i in range(len(articles_cuts[:-2]))]
words_count_2 = Counter(articles_cuts_gram2)
words_count_len = len(articles_cuts_gram2)


print(prob_2('我们', '在'))

def get_probablity(sentence):
    words = cut(sentence)
    sentence_pro = 1
    for i, word in enumerate(words[:-1]):
        next_= words[i+1]
        probability = prob_2(word, next_)
        sentence_pro *= probability
    return sentence_pro

get_probablity('小明今天抽奖抽到一台苹果手机')

words =  generate(gram=create_grammar(simple_grammar), target='sentence')
get_probablity(words)

need_compared = [
    "今天晚上请你吃大餐，我们一起吃日料 明天晚上请你吃大餐，我们一起吃苹果",
    "真事一只好看的小猫 真是一只好看的小猫",
    "今晚我去吃火锅 今晚火锅去吃我",
    "洋葱奶昔来一杯 养乐多绿来一杯"
]

for s in need_compared:
    s1, s2 = s.split()
    p1, p2 = get_probablity(s1), get_probablity(s2)
    better = s1 if p1> p2 else s2
    print('%s is better' % better)
    print('*'*40)
    print('-' * 4 + ' {} with probility {}'.format(s1, p1))
    print('-' * 4 + ' {} with probility {}'.format(s2, p2))

you_need_replace_this_with_name_you_given2 = '''
sentence =>  人物 
人物 => 男 介词 男形容词 男宾 | 女 介词 女形容词 女宾
男 => 哥们 | 兄弟 | 大叔 
女 => 姑娘 | 美女 |媳妇|大姐|阿姨
形容词 => 女形容词 | 男形容词 | 中
女形容词 => 优雅的 | 贤淑的 | 温柔的|贤惠的
男形容词 => 健壮的 | 帅气的 | 俊俏的|
中 => 善良的|大方的|文静的|脱俗的|纯洁的|开朗的|活泼的|率直的|可爱的
介词 =>  是一个| 确实是 |必须是|肯定是|当然是 |毫无疑问是
宾语 => 男宾|女宾
男宾 =>  小伙子 |男人 | 男孩 | 先生
女宾 => 小姑娘 | 大美女 |姑娘|美女|女士
'''

def generate_n(gram,target,n= 20):
    sentences = []
    for i in range(20):
        words = generate(gram=create_grammar(gram, split='=>'), target=target)
        sentences .append(words)
    # you code here
    return sentences
sentences = generate_n(you_need_replace_this_with_name_you_given2,'sentence', 20)
print(sentences)



filename = r'F:\eclipsewokspace\MachineLearing\NLP\datasource\train.txt'
content = pd.read_csv(filename,sep='',encoding='gb18030')


articless = content['content'].tolist()
articles = articless[:-1]
articles_clean = [ ''.join( token(str(a))) for a in articles]

# 这样写我电脑带不动，可能需要很大的内存吧
# words_count = Counter(reduce(add, [cut(a) for a in articles_clean]))

articles_cuts = []
for string in  articles_clean:
    articles_cuts +=cut(string)

words_count = Counter(articles_cuts)
words_count.most_common(100)
frequiences = [f for w, f in words_count.most_common(100)]
x = range(0,100)
plt.plot(x,frequiences)
plt.show()