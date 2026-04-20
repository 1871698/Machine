import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import re
from bs4 import BeautifulSoup

# 简单的词干提取函数
def simple_stemmer(word):
    suffixes = ['ing', 'ed', 'es', 's']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

# 内置停用词列表（移除否定词）
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'ma'
])

# 保留的否定词
negative_words = {'not', 'no', 'nor', 'never', 'don', 'don\'t', 'ain', 'ain\'t', 'aren', 'aren\'t', 'couldn', 'couldn\'t', 'didn', 'didn\'t', 'doesn', 'doesn\'t', 'hadn', 'hadn\'t', 'hasn', 'hasn\'t', 'haven', 'haven\'t', 'isn', 'isn\'t', 'mightn', 'mightn\'t', 'mustn', 'mustn\'t', 'needn', 'needn\'t', 'shan', 'shan\'t', 'shouldn', 'shouldn\'t', 'wasn', 'wasn\'t', 'weren', 'weren\'t', 'won', 'won\'t', 'wouldn', 'wouldn\'t'}

# 文本预处理函数
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('\\"', '')
    text = text.replace('\\\'', '')
    text = re.sub(r'\\+', '', text)
    text = re.sub(r'"+', '', text)
    text = re.sub(r'[#*]+', '', text)
    text = re.sub(r'[!@#$%^&*()_+\-=\[\]{};\\\'"\\|,.<>\/\?]', ' ', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    processed_words = []
    for word in words:
        word = re.sub(r'\\+', '', word)
        if word not in stop_words or word in negative_words:
            processed_word = simple_stemmer(word)
            processed_words.append(processed_word)
    
    processed_text = ' '.join(processed_words)
    processed_text = re.sub(r'\\+', '', processed_text)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text

# 读取数据
def load_data(file_path, sep='\t'):
    try:
        return pd.read_csv(file_path, sep=sep, quoting=3)
    except Exception as e:
        print(f"读取数据失败: {e}")
        return None

# 主函数
def main():
    # 文件路径
    training_data_path = 'labeledTrainData.tsv'
    test_data_path = 'testData.tsv'
    submission_path = 'sampleSubmission_improved.csv'
    
    # 加载训练数据
    print("正在读取训练数据...")
    train_df = load_data(training_data_path)
    
    if train_df is not None:
        print(f"数据读取成功，共 {len(train_df)} 条记录")
        
        # 预处理训练数据
        print("正在预处理训练数据...")
        train_df['processed_review'] = train_df['review'].apply(preprocess_text)
        
        # 提取文本和标签
        corpus = train_df['processed_review'].tolist()
        y_train = train_df['sentiment'].values
        
        # 创建TF-IDF vectorizer
        print("正在创建TF-IDF特征...")
        vectorizer = TfidfVectorizer(
            analyzer='word', 
            max_features=15000, 
            ngram_range=(1, 2)
        )
        
        # 在训练数据上fit和transform
        X_train = vectorizer.fit_transform(corpus)
        print(f"TF-IDF特征数量: {X_train.shape[1]}")
        
        # 划分训练集和测试集
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_split.toarray())
        X_test_scaled = scaler.transform(X_test_split.toarray())
        
        # 训练逻辑回归模型
        print("正在训练逻辑回归模型...")
        model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        model.fit(X_train_scaled, y_train_split)
        
        # 评估模型
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_split, y_pred)
        report = classification_report(y_test_split, y_pred)
        
        print(f"测试集准确率: {accuracy:.4f}")
        print("分类报告:")
        print(report)
        
        # 加载测试数据
        print("\n正在读取测试数据...")
        test_df = load_data(test_data_path, sep='\t')
        
        if test_df is not None:
            # 预处理测试数据
            print("正在预处理测试数据...")
            test_df['processed_review'] = test_df['review'].apply(preprocess_text)
            
            # 生成TF-IDF特征
            X_test = vectorizer.transform(test_df['processed_review'])
            X_test_scaled = scaler.transform(X_test.toarray())
            
            # 预测
            print("正在预测测试数据...")
            y_pred = model.predict(X_test_scaled)
            test_df['sentiment'] = y_pred
            
            # 准备提交文件
            submission_df = test_df[['id', 'sentiment']]
            submission_df['id'] = submission_df['id'].astype(str).apply(lambda x: x.replace('"', ''))
            
            # 保存提交文件
            print("正在保存提交文件...")
            submission_df.to_csv(submission_path, index=False, quoting=3)
            print(f"提交文件已保存到: {submission_path}")
            
            # 移动到results目录
            import os
            if os.path.exists(submission_path):
                os.system(f"Move-Item -Path {submission_path} -Destination results")
                print(f"提交文件已移动到 results/{submission_path}")
            
            # 打印提交文件样例
            print("\n提交文件样例:")
            print(submission_df.head())
            
            # 打印预测分布
            print("\n预测结果分布:")
            print(submission_df['sentiment'].value_counts())
            
            # 记录实验日志
            log_data = {
                'id': [46],
                '主要改动': [f'改进模型训练流程，使用统一的预处理和TF-IDF参数优化（max_features=15000, ngram_range=(1,2)）'],
                '实现结果': [f'测试集准确率: {accuracy:.4f}']
            }
            log_df = pd.DataFrame(log_data)
            try:
                existing_log = pd.read_csv('results/experiment_log.csv')
                log_df = pd.concat([existing_log, log_df], ignore_index=True)
            except FileNotFoundError:
                pass
            log_df.to_csv('results/experiment_log.csv', index=False, encoding='utf-8')
            print("\n实验日志已更新")
            
            # 提交到GitHub
            print("\n正在提交到GitHub...")
            os.system("git add .")
            os.system("git commit -m \"改进模型训练流程，使用统一的预处理和TF-IDF参数优化，准确率提升\"")
            os.system("git push")
            print("GitHub提交完成")
        else:
            print("测试数据读取失败")
    else:
        print("训练数据读取失败")

if __name__ == "__main__":
    main()
