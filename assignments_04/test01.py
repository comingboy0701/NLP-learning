# encoding: utf-8 
""" 
@author: chen km 
@contact: 760855003@qq.com 
@file: test01.py 
@time: 2019/7/28 20:34 
"""
# 在命令行中使用 
# 1.提取中文词
 WikiExtractor.py -b 1024M -o chinese zhwiki-20190720-pages-articles-multistream.xml.bz2 
 
 # 2.繁体字转为简体字
 opencc -i wiki_00 -o zh_wiki_00 -c t2s.json
 opencc -i wiki_01 -o zh_wiki_01 -c t2s.json
 
# 3. 清洗数据
 
# 4. jieba cut 
 
# python -m jieba -d " " std_zh_wiki_00 > cut_std_zh_wiki_00
# python -m jieba -d " " std_zh_wiki_02 > cut_std_zh_wiki_02
# python -m jieba -d " " std_zh_wiki_03 > cut_std_zh_wiki_03
 
# 5. traing 
 
# 6. fig
 
 
 
 