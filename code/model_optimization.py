import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec

# 读取数据
def load_data(file_path, sep='\t'):
    """读取数据文件"""
    try:
        return pd.read_csv(file_path, sep=sep, quoting=3)
    except Exception as e:
        print(f"读取数据失败: {e}")
        return None

# 加载Word2Vec模型
def load_word2vec_model(model_path):
    """加载Word2Vec模型"""
    try:
        model = Word2Vec.load(model_path)
        print(f"成功加载Word2Vec模型，词汇表大小: {len(model.wv.key_to_index)}")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

# 计算TF-IDF权重
def compute_tfidf(corpus, max_features=5000):
    """计算TF-IDF权重"""
    print(f"正在计算TF-IDF权重 (max_features={max_features})...")
    vectorizer = TfidfVectorizer(analyzer='word', max_features=max_features, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    print(f"TF-IDF特征数量: {len(feature_names)}")
    return vectorizer, tfidf_matrix, feature_names

# 生成TF-IDF加权的词向量
def get_tfidf_weighted_embedding(text, model, vectorizer, feature_names):
    """使用TF-IDF权重对词向量进行加权平均"""
    words = text.split()
    word_vectors = []
    weights = []
    
    # 计算TF-IDF权重
    tfidf_vector = vectorizer.transform([text]).toarray()[0]
    tfidf_dict = dict(zip(feature_names, tfidf_vector))
    
    for word in words:
        if word in model.wv:
            # 获取词向量
            vector = model.wv[word]
            # 获取TF-IDF权重，如果词不在特征中则使用0
            weight = tfidf_dict.get(word, 0)
            if weight > 0:
                word_vectors.append(vector)
                weights.append(weight)
    
    if word_vectors:
        # 加权平均
        weighted_vectors = np.array(word_vectors) * np.array(weights)[:, np.newaxis]
        return np.sum(weighted_vectors, axis=0) / np.sum(weights)
    else:
        # 如果没有单词在词汇表中，返回零向量
        return np.zeros(model.vector_size)

# 准备特征
def prepare_features(df, model, vectorizer, feature_names):
    """为数据准备特征"""
    print("正在生成TF-IDF加权的词向量...")
    features = []
    total = len(df)
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"处理到第 {i}/{total} 条")
        text = row.get('processed_review', '')
        embedding = get_tfidf_weighted_embedding(text, model, vectorizer, feature_names)
        features.append(embedding)
    return np.array(features)

# 准备额外特征
def prepare_additional_features(df):
    """准备额外特征"""
    print("正在准备额外特征...")
    features = []
    for _, row in df.iterrows():
        text = row.get('processed_review', '')
        words = text.split()
        # 文本长度
        text_length = len(text)
        # 单词数量
        word_count = len(words)
        # 平均单词长度
        avg_word_length = np.mean([len(word) for word in words]) if word_count > 0 else 0
        # 特征列表
        features.append([text_length, word_count, avg_word_length])
    return np.array(features)

# 训练模型
def train_model(X, y, model_type='logistic'):
    """训练不同类型的模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 根据模型类型创建模型
    if model_type == 'logistic':
        print("训练逻辑回归模型...")
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'svm':
        print("训练SVM模型...")
        model = SVC(random_state=42, kernel='linear')
    elif model_type == 'random_forest':
        print("训练随机森林模型...")
        model = RandomForestClassifier(random_state=42, n_estimators=100)
    else:
        print("未知模型类型")
        return None, 0
    
    # 训练模型
    model.fit(X_train_scaled, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"模型准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
    return model, accuracy, scaler

# 网格搜索超参数调优
def grid_search(X, y):
    """使用网格搜索优化逻辑回归模型参数"""
    print("正在进行网格搜索超参数调优...")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义参数网格
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    # 创建模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # 网格搜索
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    # 最佳参数
    best_params = grid_search.best_params_
    print(f"最佳参数: {best_params}")
    
    # 最佳模型
    best_model = grid_search.best_estimator_
    
    # 评估最佳模型
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"最佳模型准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
    return best_model, accuracy, scaler, best_params

# 生成提交文件
def generate_submission(test_df, model, scaler, word2vec_model, vectorizer, feature_names, additional_feature_cols=None, submission_path='sampleSubmission.csv'):
    """生成提交文件"""
    # 生成TF-IDF加权词向量
    X_test_word2vec = prepare_features(test_df, word2vec_model, vectorizer, feature_names)
    
    # 生成额外特征
    if additional_feature_cols:
        X_test_additional = prepare_additional_features(test_df)
        X_test = np.hstack((X_test_word2vec, X_test_additional))
    else:
        X_test = X_test_word2vec
    
    # 标准化特征
    X_test_scaled = scaler.transform(X_test)
    
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
    word2vec_model_path = 'word2vec_skipgram_model_v2.model'
    
    # 加载训练数据
    print("正在读取训练数据...")
    train_df = load_data(training_data_path)
    
    if train_df is not None:
        # 加载Word2Vec模型
        print("正在加载Word2Vec模型...")
        word2vec_model = load_word2vec_model(word2vec_model_path)
        
        if word2vec_model is not None:
            # 尝试不同的TF-IDF参数
            tfidf_max_features_list = [5000, 10000]
            
            for max_features in tfidf_max_features_list:
                print(f"\n=== 尝试 TF-IDF max_features={max_features} ===")
                # 计算TF-IDF权重
                corpus = train_df['processed_review'].tolist()
                vectorizer, tfidf_matrix, feature_names = compute_tfidf(corpus, max_features=max_features)
                
                # 准备特征
                X_word2vec = prepare_features(train_df, word2vec_model, vectorizer, feature_names)
                
                # 准备额外特征
                X_additional = prepare_additional_features(train_df)
                X_combined = np.hstack((X_word2vec, X_additional))
                
                y_train = train_df['sentiment'].values
                
                print(f"Word2Vec特征形状: {X_word2vec.shape}")
                print(f"额外特征形状: {X_additional.shape}")
                print(f"组合特征形状: {X_combined.shape}")
                
                # 尝试不同的模型
                model_types = ['logistic', 'svm', 'random_forest']
                
                for model_type in model_types:
                    print(f"\n--- 训练 {model_type} 模型 ---")
                    # 训练模型（仅使用Word2Vec特征）
                    model, accuracy, scaler = train_model(X_word2vec, y_train, model_type=model_type)
                    
                    # 记录实验日志
                    log_data = {
                        'id': [41],
                        '主要改动': [f'尝试{model_type}模型，TF-IDF max_features={max_features}，仅使用Word2Vec特征'],
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
                
                # 尝试组合特征
                print("\n--- 训练逻辑回归模型（组合特征） ---")
                model, accuracy, scaler = train_model(X_combined, y_train, model_type='logistic')
                
                # 记录实验日志
                log_data = {
                    'id': [42],
                    '主要改动': [f'尝试逻辑回归模型，TF-IDF max_features={max_features}，使用Word2Vec+额外特征'],
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
            
            # 网格搜索超参数调优
            print("\n=== 网格搜索超参数调优 ===")
            # 使用最佳参数组合
            vectorizer, tfidf_matrix, feature_names = compute_tfidf(corpus, max_features=10000)
            X_word2vec = prepare_features(train_df, word2vec_model, vectorizer, feature_names)
            X_additional = prepare_additional_features(train_df)
            X_combined = np.hstack((X_word2vec, X_additional))
            
            best_model, best_accuracy, scaler, best_params = grid_search(X_combined, y_train)
            
            # 记录实验日志
            log_data = {
                'id': [43],
                '主要改动': [f'使用网格搜索优化逻辑回归模型参数，最佳参数: {best_params}'],
                '实现结果': [f'最佳模型准确率: {best_accuracy:.4f}']
            }
            log_df = pd.DataFrame(log_data)
            # 追加到现有日志文件
            try:
                existing_log = pd.read_csv('results/experiment_log.csv')
                log_df = pd.concat([existing_log, log_df], ignore_index=True)
            except FileNotFoundError:
                pass
            log_df.to_csv('results/experiment_log.csv', index=False, encoding='utf-8')
            
            # 加载测试数据
            print("\n正在读取测试数据...")
            test_df = load_data(test_data_path, sep='\t')
            
            if test_df is not None:
                # 生成提交文件
                submission_path = 'sampleSubmission_optimized.csv'
                submission_df = generate_submission(
                    test_df, 
                    best_model, 
                    scaler, 
                    word2vec_model, 
                    vectorizer, 
                    feature_names, 
                    additional_feature_cols=True, 
                    submission_path=submission_path
                )
                
                # 移动到results目录
                import os
                if os.path.exists(submission_path):
                    os.system(f"Move-Item -Path {submission_path} -Destination results")
                    print(f"提交文件已移动到 results/{submission_path}")
                
                # 打印提交文件样例
                print("\n提交文件样例:")
                print(submission_df.head())
            else:
                print("测试数据读取失败")
        else:
            print("Word2Vec模型加载失败")
    else:
        print("训练数据读取失败")

if __name__ == "__main__":
    main()
