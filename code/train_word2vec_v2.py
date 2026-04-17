import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import time

# 读取数据
def load_data(file_path):
    """读取处理后的数据文件"""
    try:
        return pd.read_csv(file_path, sep='\t', quoting=3)
    except Exception as e:
        print(f"读取数据失败: {e}")
        return None

# 准备句子数据
def prepare_sentences(df):
    """将评论转换为句子列表，每个句子是单词列表"""
    print("正在准备句子数据...")
    sentences = []
    total = len(df)
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"处理到第 {i}/{total} 条")
        # 使用处理后的评论
        review = row.get('processed_review', '')
        if review:
            # 分词
            words = review.split()
            if len(words) > 0:
                sentences.append(words)
    print(f"句子准备完成，共 {len(sentences)} 个句子")
    return sentences

# 训练Word2Vec模型
def train_word2vec(sentences, params):
    """训练Word2Vec模型"""
    print("\n开始训练Word2Vec模型...")
    print(f"模型参数: {params}")
    
    start_time = time.time()
    
    # 创建并训练模型
    model = Word2Vec(
        sentences=sentences,
        vector_size=params['vector_size'],
        window=params['window'],
        min_count=params['min_count'],
        workers=params['workers'],
        sg=params['sg'],  # 1 for Skip-gram
        epochs=params['epochs']
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n模型训练完成，用时: {training_time:.2f} 秒")
    print(f"词汇表大小: {len(model.wv.key_to_index)}")
    
    return model, training_time

# 保存模型
def save_model(model, model_path):
    """保存Word2Vec模型"""
    print(f"\n正在保存模型到: {model_path}")
    model.save(model_path)
    print("模型保存完成")

# 测试模型
def test_model(model):
    """测试模型的基本功能"""
    print("\n测试模型功能...")
    
    # 测试词汇表
    print(f"词汇表大小: {len(model.wv.key_to_index)}")
    
    # 测试相似词
    try:
        # 随机选择几个词进行测试
        test_words = ['good', 'bad', 'great', 'movie', 'film']
        for word in test_words:
            if word in model.wv:
                similar_words = model.wv.most_similar(word, topn=5)
                print(f"与 '{word}' 相似的词: {similar_words}")
            else:
                print(f"词 '{word}' 不在词汇表中")
    except Exception as e:
        print(f"测试模型失败: {e}")

# 主函数
def main():
    # 文件路径
    processed_data_path = 'processedTrainData.tsv'
    model_path = 'word2vec_skipgram_model_v2.model'
    
    # 加载处理后的数据
    print("正在读取处理后的数据...")
    df = load_data(processed_data_path)
    
    if df is not None:
        print(f"数据读取成功，共 {len(df)} 条记录")
        
        # 准备句子数据
        sentences = prepare_sentences(df)
        
        if sentences:
            # 设置Word2Vec参数
            params = {
                'vector_size': 300,  # 词向量维度
                'window': 10,        # 窗口大小
                'min_count': 5,       # 最小词频
                'workers': 4,         # 工作线程数
                'sg': 1,              # 1 for Skip-gram
                'epochs': 15          # 训练轮数
            }
            
            # 训练模型
            model, training_time = train_word2vec(sentences, params)
            
            # 测试模型
            test_model(model)
            
            # 保存模型
            save_model(model, model_path)
            
            # 生成实验日志
            log_data = {
                'id': [39],
                '主要改动': [f'使用Skip-gram架构训练Word2Vec模型，参数：vector_size={params["vector_size"]}, window={params["window"]}, min_count={params["min_count"]}, workers={params["workers"]}, epochs={params["epochs"]}'],
                '实现结果': [f'成功训练Word2Vec模型，词汇表大小：{len(model.wv.key_to_index)}，训练用时：{training_time:.2f}秒，保存为 {model_path}']
            }
            log_df = pd.DataFrame(log_data)
            
            # 追加到现有日志文件
            try:
                existing_log = pd.read_csv('results/experiment_log.csv')
                log_df = pd.concat([existing_log, log_df], ignore_index=True)
            except FileNotFoundError:
                pass
            
            log_df.to_csv('results/experiment_log.csv', index=False, encoding='utf-8')
            print("\n实验日志已更新到 results/experiment_log.csv")
            
            print("\nWord2Vec模型训练完成！")
        else:
            print("没有准备好的句子数据，无法训练模型")
    else:
        print("数据读取失败，无法训练模型")

if __name__ == "__main__":
    main()
