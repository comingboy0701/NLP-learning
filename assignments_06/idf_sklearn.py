from sklearn.feature_extraction.text import TfidfVectorizer
 
train_document = ["The flowers are beautiful.","The name of these flowers is rose, they are very beautiful.", "Rose is beautiful", "Are you like these flowers?"]  
test_document = ["The flowers are mine.", "My flowers are beautiful"]               
 
#利用函数获取文档的TFIDF值
print("计算TF-IDF权重")       
transformer = TfidfVectorizer(max_features=5)
X_train = transformer.fit_transform(train_document)
X_test = transformer.transform(test_document)
 
#观察各个值
#（1）统计词列表
word_list = transformer.get_feature_names()  # 所有统计的词
print("统计词列表")
print(word_list)
 
#（2）统计词字典形式
print("统计词字典形式")
print(transformer.fit(test_document).vocabulary_)
 
#（3）TFIDF权重
weight_train = X_train.toarray()
weight_test = X_test.toarray()
print("train TFIDF权重")
print(weight_train)
print("test TFIDF权重")
print(weight_test)    
 
#(4)查看逆文档率（IDF）
print("train idf")
print(transformer.fit(train_document).idf_)
