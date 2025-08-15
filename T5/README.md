# T5中文文本摘要与关键词提取项目

## 项目简介
本项目基于T5模型实现中文评论的自动摘要与关键词提取，支持数据清洗、关键词抽取、T5微调、摘要生成、去重优化等完整流程。

## 主要功能
- 评论数据清洗与预处理
- 基于jieba的关键词提取
- T5模型微调与训练
- 自动生成摘要并去除重复词语
- 结果对比与可视化

## 目录结构
- `t5_finetune.py`：主程序，包含数据处理、模型训练、摘要生成等流程
- `remove_duplicates.py`：摘要去重函数
- `data_cleaning.py`：数据清洗脚本
- `cleaned_comments.csv`：清洗后的评论数据
- `train.csv` / `test.csv`：训练与测试集
- `summary_compare.csv`：模型摘要与关键词摘要对比结果
- `t5_finetuned/`：微调后模型及分词器
- `stopwords.txt`：停用词表

## 环境依赖
- Python 3.7+
- torch
- transformers
- jieba
- scikit-learn
- pandas

安装依赖：
```bash
pip install torch transformers jieba scikit-learn pandas
```

## 使用方法
1. **数据准备与清洗**
   - 将原始评论数据放入`comments.csv`，运行`data_cleaning.py`生成`cleaned_comments.csv`。
2. **关键词提取与数据集构建**
   - 运行`t5_finetune.py`，自动完成关键词提取、训练集/测试集划分。
3. **模型训练与微调**
   - 在`t5_finetune.py`中配置模型参数，运行脚本进行T5微调。
4. **摘要生成与去重**
   - 微调完成后，脚本会自动对测试集生成摘要，并调用`remove_duplicate_words`函数去除重复词。
5. **结果输出**
   - 生成的摘要与关键词对比结果保存在`summary_compare.csv`。

## 去重函数说明
摘要生成后自动调用`remove_duplicate_words`，可有效去除重复词语，提升摘要质量。



        