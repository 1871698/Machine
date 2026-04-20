import pandas as pd
import re
import os
from gensim.models import Word2Vec

# 读取processedTrainData.tsv文件
def load_processed_data(file_path):
    """读取处理后的数据文件"""
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        return None
    df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
    return df

# 准备训练数据
def prepare_training_data(df):
    """准备训练数据，将每个评论转换为单词列表"""
    sentences = []
    total_reviews = len(df)
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"处理到第 {i} 条评论")
        text = row.get('processed_review', '')
        # 将文本拆分为单词
        words = text.split()
        if words:
            sentences.append(words)
    return sentences

# 训练Word2Vec模型（调整参数）
def train_word2vec(sentences, vector_size=300, window=10, min_count=40):
    """训练Word2Vec模型"""
    print(f"\n开始训练Word2Vec模型...")
    print(f"训练数据包含 {len(sentences)} 个句子")
    print(f"参数设置: vector_size={vector_size}, window={window}, min_count={min_count}")
    
    # 设置模型参数
    model = Word2Vec(
        sentences=sentences,
        sg=1,  # 架构选择skip-gram
        hs=1,  # 训练算法：分层softmax
        sample=1e-3,  # 高频词降采样
        vector_size=vector_size,  # 词向量纬度
        window=window,  # 上下文/窗口大小
        workers=4,  # 工作线程数
        min_count=min_count  # 最小词数
    )
    
    # 调用init_sims节省内存
    model.init_sims(replace=True)
    
    print("模型训练完成")
    print(f"词汇表大小: {len(model.wv.key_to_index)}")
    
    return model

# 保存模型
def save_model(model, model_name):
    """保存模型"""
    model_path = f"{model_name}.model"
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存词向量
    vectors_path = f"{model_name}.vectors"
    model.wv.save(vectors_path)
    print(f"词向量已保存到: {vectors_path}")

# 主函数
def main():
    # 文件路径
    processed_data_path = 'processedTrainData.tsv'
    
    # 读取处理后的数据
    print("正在读取processedTrainData.tsv文件...")
    df = load_processed_data(processed_data_path)
    
    if df is not None:
        # 准备训练数据
        print("\n正在准备训练数据...")
        sentences = prepare_training_data(df)
        
        # 训练不同参数的Word2Vec模型
        # 模型1: 增加向量维度到400
        print("\n=== 训练模型1: 向量维度400, 窗口大小10 ===")
        model1 = train_word2vec(sentences, vector_size=400, window=10)
        save_model(model1, "word2vec_model_400d")
        
        # 模型2: 增加窗口大小到15
        print("\n=== 训练模型2: 向量维度300, 窗口大小15 ===")
        model2 = train_word2vec(sentences, vector_size=300, window=15)
        save_model(model2, "word2vec_model_15window")
        
        # 模型3: 同时调整向量维度和窗口大小
        print("\n=== 训练模型3: 向量维度400, 窗口大小15 ===")
        model3 = train_word2vec(sentences, vector_size=400, window=15)
        save_model(model3, "word2vec_model_400d_15window")
        
        # 写入实验日志
        print("\n正在写入实验日志...")
        log_data = {
            'id': [30],
            '主要改动': ['调整Word2Vec模型参数，训练了三个不同参数的模型：400维向量、15窗口大小、400维+15窗口'],
            '实现结果': ['成功训练了三个不同参数的Word2Vec模型，分别保存为word2vec_model_400d.model、word2vec_model_15window.model和word2vec_model_400d_15window.model']
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
    else:
        print("无法读取数据文件，实验失败")

if __name__ == "__main__":
    main()
