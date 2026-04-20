import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# 读取数据
def load_data(file_path, sep='\t'):
    """读取数据文件"""
    return pd.read_csv(file_path, sep=sep)

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

# 生成句子的均值embedding
def get_sentence_embedding(sentence, model):
    """获取句子的均值embedding"""
    words = sentence.split()
    word_vectors = []
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        # 如果没有单词在词汇表中，返回零向量
        return np.zeros(model.vector_size)

# 准备Word2Vec特征
def prepare_word2vec_features(df, model):
    """为数据准备Word2Vec特征"""
    print("正在生成Word2Vec embedding...")
    embeddings = []
    total = len(df)
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"处理到第 {i}/{total} 条")
        text = row.get('review', '')
        embedding = get_sentence_embedding(text, model)
        embeddings.append(embedding)
    return np.array(embeddings)

# 准备TF-IDF特征
def prepare_tfidf_features(df, max_features=5000):
    """为数据准备TF-IDF特征"""
    print("正在生成TF-IDF特征...")
    vectorizer = TfidfVectorizer(analyzer='word', max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['review'])
    return X.toarray(), vectorizer

# 结合Word2Vec和TF-IDF特征
def combine_features(word2vec_features, tfidf_features):
    """结合Word2Vec和TF-IDF特征"""
    print("正在结合特征...")
    return np.hstack((word2vec_features, tfidf_features))

# 训练线性回归模型
def train_linear_regression(X, y):
    """训练线性回归模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建并训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred_continuous = model.predict(X_test)
    y_pred = (y_pred_continuous > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred_continuous)
    r2 = r2_score(y_test, y_pred_continuous)
    report = classification_report(y_test, y_pred)
    
    print(f"线性回归模型准确率: {accuracy:.4f}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R2): {r2:.4f}")
    print("分类报告:")
    print(report)
    
    return model

# 生成提交文件
def generate_submission(test_df, model, word2vec_model, tfidf_vectorizer, submission_path):
    """生成提交文件"""
    # 生成Word2Vec特征
    X_test_word2vec = prepare_word2vec_features(test_df, word2vec_model)
    
    # 生成TF-IDF特征
    X_test_tfidf = tfidf_vectorizer.transform(test_df['review']).toarray()
    
    # 结合特征
    X_test = combine_features(X_test_word2vec, X_test_tfidf)
    
    # 预测
    print("正在预测测试数据...")
    y_pred_continuous = model.predict(X_test)
    test_df['sentiment'] = (y_pred_continuous > 0.5).astype(int)
    
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
    word2vec_model_path = 'word2vec_model_400d_15window_cleaned.model'
    
    # 加载训练数据
    print("正在读取训练数据...")
    train_df = load_data(training_data_path)
    
    # 加载Word2Vec模型
    print("正在加载Word2Vec模型...")
    word2vec_model = load_word2vec_model(word2vec_model_path)
    
    if word2vec_model is not None:
        # 准备Word2Vec特征
        X_train_word2vec = prepare_word2vec_features(train_df, word2vec_model)
        
        # 准备TF-IDF特征
        X_train_tfidf, tfidf_vectorizer = prepare_tfidf_features(train_df)
        
        # 结合特征
        X_train = combine_features(X_train_word2vec, X_train_tfidf)
        y_train = train_df['sentiment'].values
        
        print(f"Word2Vec特征形状: {X_train_word2vec.shape}")
        print(f"TF-IDF特征形状: {X_train_tfidf.shape}")
        print(f"组合特征形状: {X_train.shape}")
        
        # 训练线性回归模型
        print("\n正在训练线性回归模型...")
        lr_model = train_linear_regression(X_train, y_train)
        
        # 加载测试数据
        print("\n正在读取测试数据...")
        test_df = load_data(test_data_path, sep='\t')
        
        # 生成提交文件
        submission_df = generate_submission(test_df, lr_model, word2vec_model, tfidf_vectorizer, submission_path)
        
        # 写入实验日志
        print("\n正在写入实验日志...")
        log_data = {
            'id': [36],
            '主要改动': ['将逻辑回归改为线性回归，结合Word2Vec（400维，15窗口）和TF-IDF特征'],
            '实现结果': ['成功使用线性回归模型结合Word2Vec和TF-IDF特征，生成新的提交文件']
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
        print("无法加载Word2Vec模型，实验失败")

if __name__ == "__main__":
    main()
