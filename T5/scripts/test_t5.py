import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from remove_duplicates import remove_duplicate_words

if __name__ == "__main__":
    model_dir = './model/t5_finetuned'
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to('cpu')
    test_df = pd.read_csv('data/test.csv')
    import random
    # samples = test_df.sample(10, random_state=42)
    samples = test_df.sample(10)  # 每次随机采样不同
    results = []
    for _, row in samples.iterrows():
        input_text = row['input']
        gt = row['target']
        input_ids = tokenizer(input_text, return_tensors='pt', max_length=64, truncation=True).input_ids
        input_ids = input_ids.to('cpu')
        model = model.to('cpu')
        with torch.no_grad():
            output = model.generate(input_ids, max_length=20, num_beams=4)
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = remove_duplicate_words(pred)
        results.append({'原始评论': row['comment'],'模型摘要': pred, '关键词摘要': gt})
    result_df = pd.DataFrame(results)
    # 保存为 markdown 表格文件
    with open('results/summary_compare.md', 'w', encoding='utf-8') as f:
        f.write(result_df.to_markdown(index=False))
    result_df.to_csv('results/summary_compare.csv', index=False, encoding='utf-8-sig')
    print('已保存模型摘要与关键词摘要对比表 results/summary_compare.csv')
    print('已保存Markdown表格 results/summary_compare.md')