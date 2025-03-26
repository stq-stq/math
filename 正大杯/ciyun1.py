import warnings
import jieba
import jieba.analyse

import re
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup as bs
from wordcloud import WordCloud, STOPWORDS

# 忽略警告
warnings.filterwarnings("ignore")

# 设置matplotlib图形大小
plt.rcParams['figure.figsize'] = (10.0, 5.0)


# 分析网页函数
def getNowPlayingMovieList():
    url = 'https://movie.douban.com/nowplaying/guangzhou'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0'
    }
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()  # 检查请求是否成功
        html = resp.text
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP错误: {errh}")
        return []
    except requests.exceptions.RequestException as err:
        print(f"请求错误: {err}")
        return []
    soup = bs(html, 'html.parser')
    nowplaying_movie = soup.find('div', id='nowplaying')
    if not nowplaying_movie:
        return []
    nowplaying_movie_list = nowplaying_movie.find_all('li', class_='list-item')
    nowplaying_list = []
    for item in nowplaying_movie_list:
        nowplaying_dict = {}
        nowplaying_dict['id'] = item['data-subject']
        nowplaying_dict['name'] = item.find('img')['alt']
        nowplaying_list.append(nowplaying_dict)
    return nowplaying_list


# 爬取评论函数
def getCommentsById(movieId, pageNum):
    eachCommentList = []
    if pageNum <= 0:
        return eachCommentList
    start = (pageNum - 1) * 20
    url = f'https://movie.douban.com/subject/{movieId}/comments?start={start}&limit=20'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0'
    }
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()  # 检查请求是否成功
        html = resp.text
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP错误: {errh}")
        return []
    except requests.exceptions.RequestException as err:
        print(f"请求错误: {err}")
        return []
    soup = bs(html, 'html.parser')
    comment_div_lits = soup.find_all('div', class_='comment')
    for item in comment_div_lits:
        if item.find('p'):
            eachCommentList.append(item.find('p').text.strip())
    return eachCommentList


def main():
    NowPlayingMovie_list = getNowPlayingMovieList()
    if not NowPlayingMovie_list:
        print("没有获取到电影列表")
        return

    commentList = []
    for i in range(1, 11):  # 从第1页到第10页
        comments_temp = getCommentsById(NowPlayingMovie_list[0]['id'], i)  # 选择第几个电影来进行爬虫，[0]为第一个
        commentList.extend(comments_temp)

    comments = " ".join(commentList)
    # 使用正则表达式去掉标点符号和非中文字符
    pattern = re.compile(r'[^\w\s]')
    cleaned_comments = pattern.sub('', comments)

    # 使用jieba分词进行中文分词
    result = jieba.analyse.textrank(cleaned_comments, topK=150, withWeight=True)
    keywords = {word: weight for word, weight in result}

    # 停用词集合
    stopwords = set(STOPWORDS)
    with open('./StopWords.txt', encoding="utf-8") as f:
        stopwords.update(word.strip() for word in f)

    # 过滤停用词
    keywords = {word: score for word, score in keywords.items() if word not in stopwords}

    # 创建词云
    wordcloud = WordCloud(font_path="simhei.ttf", background_color="white",
                          max_font_size=80,
                          stopwords=stopwords).generate_from_frequencies(keywords)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    print('词云展示成功!')


if __name__ == "__main__":
    main()