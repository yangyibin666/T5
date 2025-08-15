from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv('data/cleaned_comments.csv')
# tokens列为分词后的list，需转为字符串
texts = df['tokens'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')

# TF-IDF向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# 标准化
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

# KMeans聚类
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df['cluster'] = labels

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled.toarray())

# 可视化
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i}')
plt.legend()
plt.title('KMeans聚类PCA降维可视化')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('results/cluster_scatter.png')
plt.show()


# 每类高频词、评分分布、情感倾向统计
def cluster_stats(df, n_top=10):
    for i in range(n_clusters):
        print(f'\n--- Cluster {i} ---')
        sub = df[df['cluster'] == i]
        # 高频词
        all_words = []
        for tokens in sub['tokens']:
            if isinstance(tokens, str):
                all_words.extend(eval(tokens))
        word_freq = Counter(all_words)
        top_words = [w for w, _ in word_freq.most_common(5)]
        print('高频词:', '、'.join(top_words))
        # 评分分布
        rating_counts = sub['rating'].value_counts().sort_index()
        print('评分分布:', dict(rating_counts))
        # 主评分段
        main_rating = rating_counts.idxmax() if not rating_counts.empty else '无'
        print(f'主评分段: {main_rating}星')
        # 简单情感倾向
        pos = sub[sub['rating'].astype(str).isin(['4', '5'])].shape[0]
        neg = sub[sub['rating'].astype(str).isin(['1', '2'])].shape[0]
        neu = sub[sub['rating'].astype(str) == '3'].shape[0]
        print(f'情感倾向: 正面{pos} 中性{neu} 负面{neg}')


if __name__ == "__main__":
    cluster_stats(df)
    print("已保存聚类散点图为cluster_scatter.png")
