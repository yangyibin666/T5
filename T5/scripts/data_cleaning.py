import pandas as pd
import jieba
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# 读取停用词表
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f if line.strip()])
    return stopwords


# 只保留中文和常用标点
def filter_chinese(text):
    return re.sub(r"[^\u4e00-\u9fa5，。！？、；：“”‘’（）()《》【】\[\]——……,.!?;:'\"()\-]", '', text)


# 分词并去除停用词
def tokenize(text, stopwords):
    words = jieba.lcut(text)
    return [w for w in words if w not in stopwords and w.strip()]


def main():
    df = pd.read_csv('data/comments.csv', on_bad_lines='skip')
    print(f"原始数据量: {len(df)}")
    # 删除缺失值
    df = df.dropna(subset=['user', 'comment'])
    # 去重
    df = df.drop_duplicates(subset=['comment'])
    # 只保留中文和标点
    df['comment'] = df['comment'].astype(str).apply(filter_chinese)
    # 再次去除空评论
    df = df[df['comment'].str.strip() != '']
    print(f"清洗后数据量: {len(df)}")
    # 加载停用词
    stopwords = load_stopwords('data/stopwords.txt')
    # 分词并去停用词
    df['tokens'] = df['comment'].apply(lambda x: tokenize(x, stopwords))
    # 保存清洗结果
    df.to_csv('data/cleaned_comments.csv', index=False, encoding='utf-8-sig')
    print("已保存清洗后数据到data/cleaned_comments.csv")

    # 生成词云图
    all_words = ' '.join([' '.join(tokens) for tokens in df['tokens']])
    wc = WordCloud(font_path='/System/Library/Fonts/STHeiti Medium.ttc', width=800, height=400,
                   background_color='white').generate(all_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/wordcloud.png', dpi=200)
    plt.show()
    print("词云图已保存到results/wordcloud.png")


if __name__ == "__main__":
    main()
