import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 读取数据
def load_data(file_path, sep='\t'):
    """读取数据文件"""
    try:
        return pd.read_csv(file_path, sep=sep, quoting=3)
    except Exception as e:
        print(f"读取数据失败: {e}")
        return None

# 计算TF-IDF权重（使用短语模式）
def compute_tfidf(corpus, max_features=5000, ngram_range=(1, 2)):
    """计算TF-IDF权重，使用短语模式"""
    print(f"正在计算TF-IDF权重（短语模式，ngram_range={ngram_range}）...")
    vectorizer = TfidfVectorizer(
        analyzer='word', 
        max_features=max_features, 
        ngram_range=ngram_range
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    print(f"TF-IDF特征数量: {len(feature_names)}")
    print(f"前10个特征: {feature_names[:10]}")
    return vectorizer, tfidf_matrix, feature_names

# 训练逻辑回归模型
def train_logistic_regression(X, y):
    """训练逻辑回归模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建并训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"逻辑回归模型准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
    return model, accuracy, scaler

# 生成提交文件
def generate_submission(test_df, model, scaler, vectorizer, submission_path):
    """生成提交文件"""
    # 生成TF-IDF特征
    X_test = vectorizer.transform(test_df['review'])
    X_test_scaled = scaler.transform(X_test.toarray())
    
    # 预测
    print("正在预测测试数据...")
    y_pred = model.predict(X_test_scaled)
    test_df['sentiment'] = y_pred
    
    # 准备提交文件
    submission_df = test_df[['id', 'sentiment']]
    
    # 移除ID列中的引号
    submission_df['id'] = submission_df['id'].astype(str).apply(lambda x: x.replace('"', ''))
    
    # 保存提交文件
    print("正在保存提交文件...")
    submission_df.to_csv(submission_path, index=False, quoting=3)
    print(f"提交文件已保存到: {submission_path}")
    
    return submission_df

# 主函数
def main():
    # 文件路径
    training_data_path = 'processedTrainData.tsv'
    test_data_path = 'testData.tsv'
    submission_path = 'sampleSubmission_logistic_ngram.csv'
    
    # 加载训练数据
    print("正在读取训练数据...")
    train_df = load_data(training_data_path)
    
    if train_df is not None:
        print(f"数据读取成功，共 {len(train_df)} 条记录")
        
        # 提取文本数据
        corpus = train_df['processed_review'].tolist()
        y_train = train_df['sentiment'].values
        
        # 尝试不同的n-gram范围
        ngram_ranges = [(1, 2), (1, 3)]
        max_features_list = [5000, 10000]
        
        best_accuracy = 0
        best_model = None
        best_scaler = None
        best_vectorizer = None
        best_params = {}
        
        for ngram_range in ngram_ranges:
            for max_features in max_features_list:
                print(f"\n=== 尝试 ngram_range={ngram_range}, max_features={max_features} ===")
                
                # 计算TF-IDF权重
                vectorizer, tfidf_matrix, feature_names = compute_tfidf(
                    corpus, 
                    max_features=max_features, 
                    ngram_range=ngram_range
                )
                
                # 转换为数组
                X_train = tfidf_matrix.toarray()
                print(f"特征形状: {X_train.shape}")
                
                # 训练模型
                model, accuracy, scaler = train_logistic_regression(X_train, y_train)
                
                # 记录最佳模型
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_scaler = scaler
                    best_vectorizer = vectorizer
                    best_params = {
                        'ngram_range': ngram_range,
                        'max_features': max_features
                    }
                
                # 记录实验日志
                log_data = {
                    'id': [44],
                    '主要改动': [f'使用逻辑回归模型，TF-IDF短语模式 ngram_range={ngram_range}, max_features={max_features}'],
                    '实现结果': [f'模型准确率: {accuracy:.4f}']
                }
                log_df = pd.DataFrame(log_data)
                # 追加到现有日志文件
                try:
                    existing_log = pd.read_csv('results/experiment_log.csv')
                    log_df = pd.concat([existing_log, log_df], ignore_index=True)
                except FileNotFoundError:
                    pass
                log_df.to_csv('results/experiment_log.csv', index=False, encoding='utf-8')
        
        # 打印最佳模型信息
        print(f"\n=== 最佳模型 ===")
        print(f"准确率: {best_accuracy:.4f}")
        print(f"参数: {best_params}")
        
        # 加载测试数据
        print("\n正在读取测试数据...")
        test_df = load_data(test_data_path, sep='\t')
        
        if test_df is not None:
            # 生成提交文件
            submission_df = generate_submission(
                test_df, 
                best_model, 
                best_scaler, 
                best_vectorizer, 
                submission_path
            )
            
            # 移动到results目录
            import os
            if os.path.exists(submission_path):
                os.system(f"Remove-Item -Path results/{os.path.basename(submission_path)} -Force -ErrorAction SilentlyContinue")
                os.system(f"Move-Item -Path {submission_path} -Destination results")
                print(f"提交文件已移动到 results/{submission_path}")
            
            # 记录最佳模型实验日志
            log_data = {
                'id': [45],
                '主要改动': [f'最佳逻辑回归模型，TF-IDF短语模式 ngram_range={best_params["ngram_range"]}, max_features={best_params["max_features"]}'],
                '实现结果': [f'模型准确率: {best_accuracy:.4f}，生成提交文件 {submission_path}']
            }
            log_df = pd.DataFrame(log_data)
            # 追加到现有日志文件
            try:
                existing_log = pd.read_csv('results/experiment_log.csv')
                log_df = pd.concat([existing_log, log_df], ignore_index=True)
            except FileNotFoundError:
                pass
            log_df.to_csv('results/experiment_log.csv', index=False, encoding='utf-8')
            
            # 打印提交文件样例
            print("\n提交文件样例:")
            print(submission_df.head())
            
            # 提交到GitHub
            print("\n正在提交到GitHub...")
            os.system("git add .")
            os.system(f"git commit -m '使用逻辑回归模型，TF-IDF短语模式 ngram_range={best_params["ngram_range"]}, max_features={best_params["max_features"]}，准确率{best_accuracy:.4f}，生成新的提交文件'")
            os.system("git push")
            print("GitHub提交完成")
        else:
            print("测试数据读取失败")
    else:
        print("训练数据读取失败")

if __name__ == "__main__":
    main()
