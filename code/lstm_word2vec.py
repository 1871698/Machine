import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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

# 准备数据
def prepare_data(df, max_length=100):
    """准备数据，将文本转换为序列"""
    print("正在准备数据...")
    
    # 分词
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['review'])
    sequences = tokenizer.texts_to_sequences(df['review'])
    
    # 填充序列
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # 标签
    labels = df['sentiment'].values
    
    return padded_sequences, labels, tokenizer

# 创建嵌入矩阵
def create_embedding_matrix(word2vec_model, tokenizer, embedding_dim):
    """创建嵌入矩阵"""
    print("正在创建嵌入矩阵...")
    
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    
    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]
    
    return embedding_matrix

# 创建LSTM模型
def create_lstm_model(embedding_matrix, max_length, embedding_dim):
    """创建LSTM模型"""
    model = Sequential()
    
    # 嵌入层
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False
    ))
    
    # LSTM层
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    
    # 输出层
    model.add(Dense(1, activation='sigmoid'))
    
    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# 训练模型
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """训练模型"""
    print("正在训练LSTM模型...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return history

# 评估模型
def evaluate_model(model, X_test, y_test):
    """评估模型"""
    print("正在评估模型...")
    
    # 预测
    y_pred = model.predict(X_test) > 0.5
    y_pred = y_pred.astype(int).flatten()
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"LSTM模型准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
    return accuracy

# 生成提交文件
def generate_submission(test_df, model, tokenizer, max_length, submission_path):
    """生成提交文件"""
    # 准备测试数据
    print("正在准备测试数据...")
    sequences = tokenizer.texts_to_sequences(test_df['review'])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # 预测
    print("正在预测测试数据...")
    y_pred = model.predict(padded_sequences) > 0.5
    test_df['sentiment'] = y_pred.astype(int).flatten()
    
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
    
    # 加载测试数据
    print("正在读取测试数据...")
    test_df = load_data(test_data_path, sep='\t')
    
    # 加载Word2Vec模型
    print("正在加载Word2Vec模型...")
    word2vec_model = load_word2vec_model(word2vec_model_path)
    
    if word2vec_model is not None:
        # 准备数据
        max_length = 100
        embedding_dim = word2vec_model.vector_size
        
        X, y, tokenizer = prepare_data(train_df, max_length)
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 创建嵌入矩阵
        embedding_matrix = create_embedding_matrix(word2vec_model, tokenizer, embedding_dim)
        
        # 创建LSTM模型
        model = create_lstm_model(embedding_matrix, max_length, embedding_dim)
        print(model.summary())
        
        # 训练模型
        train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
        
        # 评估模型
        evaluate_model(model, X_val, y_val)
        
        # 生成提交文件
        submission_df = generate_submission(test_df, model, tokenizer, max_length, submission_path)
        
        # 写入实验日志
        print("\n正在写入实验日志...")
        log_data = {
            'id': [34],
            '主要改动': ['使用Word2Vec嵌入和LSTM模型训练情感分析模型'],
            '实现结果': ['成功训练LSTM模型，使用Word2Vec嵌入，生成新的提交文件']
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
