# 去除摘要中重复词语的函数
def remove_duplicate_words(summary):
    words = summary.split()
    seen = set()
    result = []
    for word in words:
        if word not in seen:
            result.append(word)
            seen.add(word)
    return ' '.join(result)
