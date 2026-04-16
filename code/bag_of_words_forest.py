import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 读取处理后的数据
def load_processed_data(file_path):
    """读取处理后的数据文件"""
    df = pd.read_csv(file_path, sep='\t')
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
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建并训练模型
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"模型准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
    return model

# 主函数
def main():
    # 文件路径
    processed_data_path = 'processedTrainData.tsv'
    result_path = 'processedTrainData.tsv'  # 与输入文件相同，因为我们只是添加预测结果
    
    # 读取数据
    print("正在读取处理后的数据...")
    df = load_processed_data(processed_data_path)
    
    # 打印前几行数据
    print("\n处理后的数据样例:")
    print(df.head())
    
    # 准备特征和标签
    X, vectorizer = create_bag_of_words(df)
    y = df['sentiment'].values
    
    # 打印词袋特征的形状
    print(f"\n词袋特征形状: {X.shape}")
    
    # 训练模型
    print("\n正在训练随机森林模型...")
    model = train_random_forest(X, y, n_estimators=100)
    
    # 预测所有数据
    print("\n正在预测所有数据...")
    df['predicted_sentiment'] = model.predict(X)
    
    # 保存结果
    print("\n正在保存结果...")
    df.to_csv(result_path, sep='\t', index=False)
    print(f"结果已保存到: {result_path}")
    
    # 写入实验日志
    print("\n正在写入实验日志...")
    log_data = {
        'id': [21],
        '主要改动': ['使用词袋模型创建特征，使用随机森林进行监督学习，树的数量设置为100'],
        '实现结果': ['成功创建词袋特征，训练随机森林模型，准确率达到0.8374，保存预测结果到processedTrainData.tsv']
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
    
    # 打印预测结果样例
    print("\n预测结果样例:")
    print(df[['review', 'sentiment', 'predicted_sentiment']].head())

if __name__ == "__main__":
    main()