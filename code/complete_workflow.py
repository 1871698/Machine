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
    
    # 1. 移除 HTML 标记
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. 清理转义字符和反斜杠
    text = text.replace('\\"', '')
    text = text.replace('\\\'', '')
    text = re.sub(r'\\+', '', text)
    
    # 3. 清理多余的引号
    text = re.sub(r'"+', '', text)
    
    # 4. 清理特殊符号
    text = re.sub(r'[#*]+', '', text)
    
    # 5. 清理其他特殊字符
    text = re.sub(r'[!@#$%^&*()_+\-=\[\]{};\\\'"\\|,.<>\/\?]', ' ', text)
    
    # 6. 小写化
    text = text.lower()
    
    # 7. 处理数字（移除）
    text = re.sub(r'\d+', '', text)
    
    # 8. 处理标点符号和多余的空格
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 9. 分词
    words = text.split()
    
    # 10. 移除停用词并进行词干提取
    processed_words = []
    for word in words:
        word = re.sub(r'\\+', '', word)
        if word not in stop_words or word in negative_words:
            processed_word = simple_stemmer(word)
            processed_words.append(processed_word)
    
    # 11. 将单词重新连接成一个字符串
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
    submission_path = 'sampleSubmission.csv'
    
    # 1. 读取训练数据
    print("1. 正在读取 labeledTrainData.tsv...")
    train_df = load_data(training_data_path)
    
    if train_df is not None:
        print(f"数据读取成功，共 {len(train_df)} 条记录")
        
        # 2. 文本清洗
        print("2. 正在进行文本清洗...")
        train_df['processed_review'] = train_df['review'].apply(preprocess_text)
        
        # 3. 划分训练/验证集
        print("3. 正在划分训练/验证集...")
        X_train, X_val, y_train, y_val = train_test_split(
            train_df['processed_review'], 
            train_df['sentiment'], 
            test_size=0.2, 
            random_state=42
        )
        print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        
        # 4. 使用 TF-IDF + LogisticRegression 训练
        print("4. 正在训练 TF-IDF + LogisticRegression 模型...")
        # 创建TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            analyzer='word', 
            max_features=15000, 
            ngram_range=(1, 2)
        )
        
        # 拟合训练数据
        X_train_tfidf = vectorizer.fit_transform(X_train)
        print(f"TF-IDF特征数量: {X_train_tfidf.shape[1]}")
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_tfidf.toarray())
        
        # 训练逻辑回归模型
        model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        model.fit(X_train_scaled, y_train)
        
        # 5. 在验证集评估
        print("5. 正在在验证集上评估模型...")
        X_val_tfidf = vectorizer.transform(X_val)
        X_val_scaled = scaler.transform(X_val_tfidf.toarray())
        y_pred = model.predict(X_val_scaled)
        
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        
        print(f"验证集准确率: {accuracy:.4f}")
        print("分类报告:")
        print(report)
        
        # 6. 读取 testData.tsv 并生成预测
        print("6. 正在读取 testData.tsv 并生成预测...")
        test_df = load_data(test_data_path, sep='\t')
        
        if test_df is not None:
            print(f"测试数据读取成功，共 {len(test_df)} 条记录")
            
            # 预处理测试数据
            test_df['processed_review'] = test_df['review'].apply(preprocess_text)
            
            # 生成TF-IDF特征
            X_test_tfidf = vectorizer.transform(test_df['processed_review'])
            X_test_scaled = scaler.transform(X_test_tfidf.toarray())
            
            # 预测
            y_test_pred = model.predict(X_test_scaled)
            test_df['sentiment'] = y_test_pred
            
            # 7. 按 sampleSubmission.csv 格式导出
            print("7. 正在按 sampleSubmission.csv 格式导出结果...")
            submission_df = test_df[['id', 'sentiment']]
            
            # 移除ID列中的引号
            submission_df['id'] = submission_df['id'].astype(str).apply(lambda x: x.replace('"', ''))
            
            # 保存提交文件
            submission_df.to_csv(submission_path, index=False, quoting=3)
            print(f"提交文件已保存到: {submission_path}")
            
            # 移动到results目录
            import os
            if os.path.exists(submission_path):
                os.system(f"Remove-Item -Path results/sampleSubmission.csv -Force -ErrorAction SilentlyContinue")
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
                'id': [47],
                '主要改动': ['按照要求执行完整流程：文本清洗、划分训练/验证集、TF-IDF+LogisticRegression训练、验证评估、测试预测、导出提交文件'],
                '实现结果': [f'验证集准确率: {accuracy:.4f}，生成提交文件 sampleSubmission.csv']
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
            os.system("git commit -m \"按照要求执行完整流程：文本清洗、模型训练、验证评估、测试预测、导出提交文件\"")
            os.system("git push")
            print("GitHub提交完成")
        else:
            print("测试数据读取失败")
    else:
        print("训练数据读取失败")

if __name__ == "__main__":
    main()
