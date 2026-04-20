import os
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

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

# 加载数据
def load_data(data_dir):
    train_path = os.path.join(data_dir, 'labeledTrainData.tsv')
    test_path = os.path.join(data_dir, 'testData.tsv')
    unlabeled_path = os.path.join(data_dir, 'unlabeledTrainData.tsv')
    
    train_df = pd.read_csv(train_path, sep='\t', quoting=3, on_bad_lines='skip')
    test_df = pd.read_csv(test_path, sep='\t', quoting=3, on_bad_lines='skip')
    unlabeled_df = pd.read_csv(unlabeled_path, sep='\t', quoting=3, on_bad_lines='skip')
    
    return train_df, test_df, unlabeled_df

# 训练Word2Vec模型
def train_word2vec(data):
    sentences = []
    for text in data:
        sentences.append(simple_preprocess(text))
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=300,
        window=10,
        min_count=5,
        workers=4,
        sg=1,  # Skip-gram
        epochs=15
    )
    return model

# 获取Word2Vec文档向量
def get_word2vec_embedding(text, model):
    words = simple_preprocess(text)
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data_raw', help='数据目录')
    parser.add_argument('--features', type=str, default='tfidf', choices=['tfidf', 'word2vec', 'tfidf_both'], help='特征类型')
    parser.add_argument('--out', type=str, default='./submission.csv', help='输出文件路径')
    args = parser.parse_args()
    
    print('加载数据...')
    train_df, test_df, unlabeled_df = load_data(args.data_dir)
    
    print('预处理文本...')
    train_df['cleaned_review'] = train_df['review'].apply(preprocess_text)
    test_df['cleaned_review'] = test_df['review'].apply(preprocess_text)
    unlabeled_df['cleaned_review'] = unlabeled_df['review'].apply(preprocess_text)
    
    # 合并所有数据用于Word2Vec训练
    all_texts = pd.concat([train_df['cleaned_review'], unlabeled_df['cleaned_review']])
    
    if args.features == 'word2vec':
        print('训练Word2Vec模型...')
        word2vec_model = train_word2vec(all_texts)
        
        print('生成Word2Vec特征...')
        X_train = np.array([get_word2vec_embedding(text, word2vec_model) for text in train_df['cleaned_review']])
        X_test = np.array([get_word2vec_embedding(text, word2vec_model) for text in test_df['cleaned_review']])
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        
    elif args.features == 'tfidf' or args.features == 'tfidf_both':
        if args.features == 'tfidf_both':
            # 词级和字符级TF-IDF
            word_vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            char_vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(2, 5),
                analyzer='char'
            )
            
            print('生成TF-IDF特征...')
            X_train_word = word_vectorizer.fit_transform(train_df['cleaned_review'])
            X_train_char = char_vectorizer.fit_transform(train_df['cleaned_review'])
            X_train = np.hstack([X_train_word.toarray(), X_train_char.toarray()])
            
            X_test_word = word_vectorizer.transform(test_df['cleaned_review'])
            X_test_char = char_vectorizer.transform(test_df['cleaned_review'])
            X_test = np.hstack([X_test_word.toarray(), X_test_char.toarray()])
        else:
            # 仅词级TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            print('生成TF-IDF特征...')
            X_train = vectorizer.fit_transform(train_df['cleaned_review'])
            X_test = vectorizer.transform(test_df['cleaned_review'])
        
        # 使用SGDClassifier作为SVM模型
        model = SGDClassifier(
            loss='hinge',  # SVM
            penalty='l2',
            alpha=1e-4,
            random_state=42,
            max_iter=1000,
            tol=1e-3
        )
    
    y_train = train_df['sentiment'].values
    
    print('训练模型...')
    model.fit(X_train, y_train)
    
    print('交叉验证...')
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f'交叉验证准确率: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    
    print('预测测试集...')
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.decision_function(X_test)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': y_pred
    })
    submission.to_csv(args.out, index=False, quoting=3)
    print(f'提交文件已保存至: {args.out}')

if __name__ == '__main__':
    main()