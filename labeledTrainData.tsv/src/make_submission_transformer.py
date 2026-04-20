import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from bs4 import BeautifulSoup
import re

# 文本预处理函数
def preprocess_text(text):
    # 移除HTML标签
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    # 仅保留字母
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 转小写
    text = text.lower()
    # 合并多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 数据集类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 加载数据
def load_data(data_dir):
    train_path = os.path.join(data_dir, 'labeledTrainData.tsv')
    test_path = os.path.join(data_dir, 'testData.tsv')
    
    train_df = pd.read_csv(train_path, sep='\t', quoting=3, on_bad_lines='skip')
    test_df = pd.read_csv(test_path, sep='\t', quoting=3, on_bad_lines='skip')
    
    return train_df, test_df

# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data_raw', help='数据目录')
    parser.add_argument('--model', type=str, default='roberta-large', choices=['roberta-large', 'distilbert-base-uncased'], help='模型类型')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='批量大小')
    parser.add_argument('--grad-accum', type=int, default=8, help='梯度累积步数')
    parser.add_argument('--fp16', action='store_true', help='使用混合精度训练')
    parser.add_argument('--grad-checkpointing', action='store_true', help='使用梯度检查点')
    parser.add_argument('--valid-ratio', type=float, default=0, help='验证集比例')
    parser.add_argument('--no-eval', action='store_true', help='不进行评估')
    parser.add_argument('--out', type=str, default='./submission_transformer.csv', help='输出文件路径')
    args = parser.parse_args()
    
    print('加载数据...')
    train_df, test_df = load_data(args.data_dir)
    
    print('预处理文本...')
    train_df['cleaned_review'] = train_df['review'].apply(preprocess_text)
    test_df['cleaned_review'] = test_df['review'].apply(preprocess_text)
    
    # 加载模型和分词器
    if args.model == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=2)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    # 准备数据集
    train_texts = train_df['cleaned_review'].tolist()
    train_labels = train_df['sentiment'].tolist()
    test_texts = test_df['cleaned_review'].tolist()
    
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, None, tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir='./artifacts_transformer',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        fp16=args.fp16,
        gradient_checkpointing=args.grad_checkpointing,
        evaluation_strategy='no' if args.no_eval else 'epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_dir='./logs',
        logging_steps=100,
        save_total_limit=2,
        weight_decay=0.01,
        learning_rate=2e-5,
        warmup_steps=500,
        max_steps=-1
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None if args.valid_ratio == 0 else train_dataset
    )
    
    print('训练模型...')
    trainer.train()
    
    print('预测测试集...')
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': y_pred
    })
    submission.to_csv(args.out, index=False, quoting=3)
    print(f'提交文件已保存至: {args.out}')

if __name__ == '__main__':
    main()