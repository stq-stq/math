# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
#
# # 读取文本文件
# with open('text.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
# font_path = 'C:/Windows/Fonts/simhei.ttf'
# # 创建词云对象
# wordcloud = WordCloud(font_path=font_path,background_color='white', width=800, height=400, max_font_size=100).generate(text)
#
# # 绘制词云图
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
#
# # 显示词云图
# plt.show()



import jieba.analyse

import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号





# 使用jieba进行中文分词
# 示例文本文件路径
file_path = 'text.txt'  # 请替换为你的文本文件路径
with open('text.txt', 'r', encoding='utf-8') as f:
    text12321 = f.read()
# print("text", text12321)

# words = jieba.lcut(text12321)
# 使用正则表达式去掉标点符号和非中文字符
pattern = re.compile(r'[^\w\s]')
cleaned_comments = pattern.sub('', text12321)

# 使用jieba分词进行中文分词
result = jieba.analyse.textrank(cleaned_comments, topK=150, withWeight=True)
words = {word: weight for word, weight in result}


# 统计词频
word_freq = {}
for word in words:
    if word in word_freq:
        word_freq[word] += 1
    else:
        word_freq[word] = 1

filtered_word_freq = {word: freq for word, freq in word_freq.items() if freq > 1}



# 过滤掉符号
filtered_word_freq = {re.sub(r'[^\w\s]', '', word): freq for word, freq in filtered_word_freq.items()}

# 按词频降序排列
sorted_word_freq = sorted(filtered_word_freq.items(), key=lambda x: x[1], reverse=True)

# 可视化结果（使用词云图）
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 修改为你的字体文件路径
plt.figure(figsize=(10, 5))
wordcloud = WordCloud(font_path=font_path, width=800, height=400).generate_from_frequencies(filtered_word_freq)
plt.imshow(wordcloud, interpolation='bilinear')
plt.figure(figsize=(10, 5))
plt.axis('off')
plt.title("文本词频统计")
plt.show()


