import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
from text_preprocessing import preprocess_text

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

# 准备特征
def prepare_features(df, model):
    """为数据准备特征"""
    print("正在生成句子embedding...")
    embeddings = []
    total = len(df)
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"处理到第 {i}/{total} 条")
        text = row.get('processed_review', '')
        embedding = get_sentence_embedding(text, model)
        embeddings.append(embedding)
    return np.array(embeddings)

# 训练逻辑回归模型
def train_logistic_regression(X, y):
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
    
    print(f"逻辑回归模型准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
    return model

# 生成提交文件
def generate_submission(test_df, model, word2vec_model, submission_path):
    """生成提交文件"""
    # 预处理测试数据
    print("正在预处理测试数据...")
    test_df['processed_review'] = test_df['review'].apply(preprocess_text)
    
    # 生成特征
    X_test = prepare_features(test_df, word2vec_model)
    
    # 预测
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
    
    return submission_df

# 主函数
def main():
    # 文件路径
    training_data_path = 'processedTrainData.tsv'
    test_data_path = 'testData.tsv'
    submission_path = 'sampleSubmission.csv'
    word2vec_model_path = 'skipgram_word2vec_model.model'
    
    # 加载训练数据
    print("正在读取训练数据...")
    train_df = load_data(training_data_path)
    
    # 加载Word2Vec模型
    print("正在加载Word2Vec模型...")
    word2vec_model = load_word2vec_model(word2vec_model_path)
    
    if word2vec_model is not None:
        # 准备特征
        X_train = prepare_features(train_df, word2vec_model)
        y_train = train_df['sentiment'].values
        
        print(f"特征形状: {X_train.shape}")
        
        # 训练逻辑回归模型
        print("\n正在训练逻辑回归模型...")
        lr_model = train_logistic_regression(X_train, y_train)
        
        # 加载测试数据
        print("\n正在读取测试数据...")
        test_df = load_data(test_data_path, sep='\t')
        
        # 生成提交文件
        submission_df = generate_submission(test_df, lr_model, word2vec_model, submission_path)
        
        # 写入实验日志
        print("\n正在写入实验日志...")
        log_data = {
            'id': [29],
            '主要改动': ['使用Word2Vec模型生成均值embedding，然后训练逻辑回归模型'],
            '实现结果': ['成功使用Word2Vec均值embedding和逻辑回归模型，生成新的提交文件']
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
