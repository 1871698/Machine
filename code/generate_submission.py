import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
from text_preprocessing import preprocess_text

# 读取训练数据
def load_training_data(file_path):
    """读取训练数据"""
    df = pd.read_csv(file_path, sep='\t')
    return df

# 读取测试数据
def load_test_data(file_path):
    """读取测试数据"""
    df = pd.read_csv(file_path, sep='\t', escapechar='\\', quoting=3)
    return df

# 创建词袋特征
def create_bag_of_words(df, max_features=5000):
    """使用CountVectorizer创建词袋特征"""
    vectorizer = CountVectorizer(analyzer='word', max_features=max_features)
    X = vectorizer.fit_transform(df['processed_review'])
    return X, vectorizer

# 训练随机森林模型
def train_random_forest(X, y, n_estimators=100):
    """使用随机森林进行监督学习"""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    return model

# 主函数
def main():
    # 文件路径
    training_data_path = 'processedTrainData.tsv'
    test_data_path = 'testData.tsv'
    submission_path = 'sampleSubmission.csv'
    
    # 读取训练数据
    print("正在读取训练数据...")
    train_df = load_training_data(training_data_path)
    
    # 预处理训练数据（如果需要）
    if 'processed_review' not in train_df.columns:
        print("正在预处理训练数据...")
        train_df['processed_review'] = train_df['review'].apply(preprocess_text)
        train_df.to_csv(training_data_path, sep='\t', index=False)
    
    # 准备特征和标签
    print("正在创建词袋特征...")
    X_train, vectorizer = create_bag_of_words(train_df)
    y_train = train_df['sentiment'].values
    
    # 训练模型
    print("正在训练随机森林模型...")
    model = train_random_forest(X_train, y_train, n_estimators=100)
    
    # 读取测试数据
    print("正在读取测试数据...")
    test_df = load_test_data(test_data_path)
    
    # 预处理测试数据
    print("正在预处理测试数据...")
    test_df['processed_review'] = test_df['review'].apply(preprocess_text)
    
    # 将测试数据转换为词袋特征
    print("正在转换测试数据为词袋特征...")
    X_test = vectorizer.transform(test_df['processed_review'])
    
    # 预测测试数据
    print("正在预测测试数据...")
    test_df['sentiment'] = model.predict(X_test)
    
    # 准备提交文件
    submission_df = test_df[['id', 'sentiment']]
    
    # 去除ID列中的引号
    submission_df['id'] = submission_df['id'].str.replace('"', '')
    
    # 保存提交文件
    print("正在保存提交文件...")
    submission_df.to_csv(submission_path, index=False, quoting=3)
    print(f"提交文件已保存到: {submission_path}")
    
    # 写入实验日志
    print("正在写入实验日志...")
    log_data = {
        'id': [26],
        '主要改动': ['基于词袋模型和随机森林模型生成提交文件sampleSubmission.csv'],
        '实现结果': ['成功对testData.tsv进行预处理，使用训练好的模型预测情感标签，生成符合要求格式的提交文件']
    }
    log_df = pd.DataFrame(log_data)
    # 追加到现有日志文件
    try:
        existing_log = pd.read_csv('experiment_log.csv')
        log_df = pd.concat([existing_log, log_df], ignore_index=True)
    except FileNotFoundError:
        pass
    log_df.to_csv('experiment_log.csv', index=False, encoding='utf-8')
    print("实验日志已更新")
    
    # 打印提交文件样例
    print("\n提交文件样例:")
    print(submission_df.head())

if __name__ == "__main__":
    main()
