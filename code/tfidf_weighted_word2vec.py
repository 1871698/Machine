import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
    print("正在计算TF-IDF权重...")
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

# 训练模型
def train_model(X, y):
    """训练逻辑回归模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建并训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"模型准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
    return model, accuracy

# 生成提交文件
def generate_submission(test_df, model, word2vec_model, vectorizer, feature_names, submission_path):
    """生成提交文件"""
    # 生成特征
    X_test = prepare_features(test_df, word2vec_model, vectorizer, feature_names)
    
    # 预测
    print("正在预测测试数据...")
    y_pred = model.predict(X_test)
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
    submission_path = 'sampleSubmission.csv'
    word2vec_model_path = 'word2vec_skipgram_model_v2.model'
    
    # 加载训练数据
    print("正在读取训练数据...")
    train_df = load_data(training_data_path)
    
    if train_df is not None:
        # 加载Word2Vec模型
        print("正在加载Word2Vec模型...")
        word2vec_model = load_word2vec_model(word2vec_model_path)
        
        if word2vec_model is not None:
            # 计算TF-IDF权重
            corpus = train_df['processed_review'].tolist()
            vectorizer, tfidf_matrix, feature_names = compute_tfidf(corpus)
            
            # 准备特征
            X_train = prepare_features(train_df, word2vec_model, vectorizer, feature_names)
            y_train = train_df['sentiment'].values
            
            print(f"特征形状: {X_train.shape}")
            
            # 训练模型
            print("\n正在训练逻辑回归模型...")
            lr_model, accuracy = train_model(X_train, y_train)
            
            # 加载测试数据
            print("\n正在读取测试数据...")
            test_df = load_data(test_data_path, sep='\t')
            
            if test_df is not None:
                # 生成提交文件
                submission_df = generate_submission(test_df, lr_model, word2vec_model, vectorizer, feature_names, submission_path)
                
                # 写入实验日志
                print("\n正在写入实验日志...")
                log_data = {
                    'id': [40],
                    '主要改动': ['使用TF-IDF权重对Word2Vec词向量进行加权平均，构建特征工程，训练逻辑回归模型'],
                    '实现结果': [f'成功训练模型，准确率: {accuracy:.4f}，生成提交文件 sampleSubmission.csv']
                }
                log_df = pd.DataFrame(log_data)
                # 追加到现有日志文件
                try:
                    existing_log = pd.read_csv('results/experiment_log.csv')
                    log_df = pd.concat([existing_log, log_df], ignore_index=True)
                except FileNotFoundError:
                    pass
                log_df.to_csv('results/experiment_log.csv', index=False, encoding='utf-8')
                print("实验日志已更新")
                
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
