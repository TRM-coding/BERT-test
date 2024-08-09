import re
import jieba
import torch

train_text = []
train_label = []
test_text = []
test_label = []



with open('./data/train_5.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip().split("  ")
        train_text.append(line[1])
        train_label.append(line[0])
#print(train_text[3])
#print(train_label)
train_label = [int(num) for num in train_label]
train_label =  torch.tensor(train_label)
#print(len(train_label))  # 15000
#print(train_text)

with open('./data/test_5.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip().split("  ")
        test_text.append(line[1])
        test_label.append(line[0])
#print(test_text[2])
#print(test_label[2])
test_label = [int(num) for num in test_label]
test_label = torch.tensor(test_label)
#print(len(test_label))  # 2500
#print(test_text)

'''训练集的长度'''
len_train = len(train_label)
'''测试集的长度'''
len_test = len(test_label)


'''数据预处理'''
def preprocess_chinese_text(text):
    # 去除英文字符和数字
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    # 去除中文标点符号
    text = re.sub(r'[，。！？、《》（）【】“”‘’『』%()\]\[,.!?<>{}:-=-：@#$\%^&*_+*]', '', text)
    # 去除空格和换行符
    text = re.sub(r'\s+', '', text)
    return text

train_text = [preprocess_chinese_text(text) for text in train_text]
#print(train_text)
#print(train_label)
test_text = [preprocess_chinese_text(text) for text in test_text]
#print(test_text)


'''
train_text = [[text] for text in train_text]
#print(train_text)
test_text = [[text] for text in test_text]
#print(test_text)
'''


'''分词'''
'''
# 对每个文本进行分词
train_text = [list(jieba.cut(text)) for text in train_text]
#print(train_text)
# 打印分词结果
#for i, words in enumerate(train_text):
#    print(f"文本 {i+1}: {words}")
test_text = [list(jieba.cut(text)) for text in test_text]
#print(test_text)


len_of_train_text = [len(sublist) for sublist in train_text]
max_len_of_train_text = max(len_of_train_text)
#print(max_len_of_train_text)  # 17
len_of_test_text = [len(sublist) for sublist in test_text]
max_len_of_test_text = max(len_of_test_text)
#print(max_len_of_test_text)  # 16
'''






