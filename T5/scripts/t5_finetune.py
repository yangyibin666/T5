import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import pandas as pd
import jieba.analyse
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch


# 1. 数据准备：提取关键词摘要
def extract_keywords(text, topk=3):
    return ' '.join(jieba.analyse.extract_tags(text, topK=topk))


def build_dataset():
    df = pd.read_csv('data/cleaned_comments.csv')
    df['keywords'] = df['comment'].apply(lambda x: extract_keywords(str(x)))
    df['keywords'] = df['keywords'].astype(str)
    df['input'] = 'summarize: ' + df['comment'].astype(str)
    df['target'] = df['keywords'].astype(str)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv('train.csv', index=False, encoding='utf-8-sig')
    test.to_csv('test.csv', index=False, encoding='utf-8-sig')
    return train, test


# 2. 数据集类
def load_data(filename):
    df = pd.read_csv(filename)
    df['input'] = df['input'].astype(str)
    df['target'] = df['target'].astype(str)
    return df['input'].tolist(), df['target'].tolist()


class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, token, max_input_length=64, max_target_length=20):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = token
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_enc = self.tokenizer(self.inputs[idx], max_length=self.max_input_length, truncation=True,
                                   padding='max_length', return_tensors='pt')
        target_enc = self.tokenizer(self.targets[idx], max_length=self.max_target_length, truncation=True,
                                    padding='max_length', return_tensors='pt')
        return {
            'input_ids': input_enc['input_ids'].squeeze(),
            'attention_mask': input_enc['attention_mask'].squeeze(),
            'labels': target_enc['input_ids'].squeeze()
        }


if __name__ == "__main__":
    # 1. 数据准备
    train_df, test_df = build_dataset()
    # 2. 加载模型与分词器
    model_name = 'Langboat/mengzi-t5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to('cpu')
    # 3. 构建数据集
    train_inputs, train_targets = load_data('data/train.csv')
    test_inputs, test_targets = load_data('data/test.csv')
    train_dataset = T5Dataset(train_inputs, train_targets, tokenizer)
    test_dataset = T5Dataset(test_inputs, test_targets, tokenizer)
    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir='./model/t5_finetuned',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=50,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy='epoch',
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to=[]
    )
    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    model.save_pretrained('./model/t5_finetuned')
    tokenizer.save_pretrained('./model/t5_finetuned')
