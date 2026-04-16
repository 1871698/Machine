# 机器学习实验2：情感分析

## 仓库结构

- `code/`：实验代码
  - `text_preprocessing.py`：文本预处理
  - `bag_of_words_forest.py`：词袋模型和随机森林
  - `train_word2vec.py`：Word2Vec模型训练
  - `train_word2vec_tuned.py`：调整参数的Word2Vec模型训练
  - `train_word2vec_cleaned.py`：基于清洗数据的Word2Vec模型训练
  - `generate_submission.py`：生成提交文件
  - `punkt_sentences.py`：句子拆分
  - `word2vec_sentences.py`：Word2Vec句子处理
  - `word2vec_logistic_regression.py`：Word2Vec和逻辑回归
  - `word2vec_logistic_regression_cleaned.py`：基于清洗数据的Word2Vec和逻辑回归
  - `word2vec_tfidf_combined.py`：Word2Vec和TF-IDF特征结合
  - `word2vec_tfidf_combined_cleaned.py`：基于清洗数据的Word2Vec和TF-IDF特征结合
  - `lstm_word2vec.py`：LSTM和Word2Vec模型

- `report/`：实验报告

- `results/`：实验结果
  - `experiment_log.csv`：实验日志
  - `sampleSubmission.csv`：提交文件
  - `over view.png`：结果截图

## 实验内容

1. **文本预处理**：移除HTML标记、处理标点符号、数字和停用词，保留否定词
2. **词袋模型**：使用CountVectorizer创建特征
3. **随机森林**：训练分类模型
4. **Word2Vec**：训练分布式词向量模型，调整参数（向量维度、窗口大小）
5. **逻辑回归**：使用Word2Vec均值embedding作为特征
6. **特征组合**：结合Word2Vec和TF-IDF特征
7. **深度学习**：尝试LSTM和Word2Vec模型
8. **生成提交文件**：预测测试数据并生成符合Kaggle要求的提交文件

## 模型性能对比

| 方法 | 准确率 | 特征类型 |
|------|--------|----------|
| 随机森林 + 词袋 | 0.8374 | 词频向量 |
| 逻辑回归 + 词袋 | 0.8616 | 词频向量 |
| 逻辑回归 + TF-IDF | 0.8868 | TF-IDF向量 |
| 逻辑回归 + Word2Vec (300d) | 0.8662 | 300维词向量均值 |
| 逻辑回归 + Word2Vec (400d, 15window) | 0.8708 | 400维词向量均值 |
| 逻辑回归 + Word2Vec + TF-IDF | **0.8886** | 400维词向量均值 + 5000维TF-IDF |

## 最佳模型

**逻辑回归 + Word2Vec + TF-IDF**
- 准确率：0.8886
- 特征：400维Word2Vec均值embedding + 5000维TF-IDF
- 优势：结合了Word2Vec的语义信息和TF-IDF的词频信息

## 依赖

- pandas
- numpy
- scikit-learn
- beautifulsoup4
- nltk
- gensim
