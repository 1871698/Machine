# 机器学习实验2：情感分析

## 仓库结构

- `code/`：实验代码
  - `text_preprocessing.py`：文本预处理
  - `bag_of_words_forest.py`：词袋模型和随机森林
  - `train_word2vec.py`：Word2Vec模型训练
  - `generate_submission.py`：生成提交文件
  - `punkt_sentences.py`：句子拆分
  - `word2vec_sentences.py`：Word2Vec句子处理

- `report/`：实验报告

- `results/`：实验结果
  - `experiment_log.csv`：实验日志
  - `sampleSubmission.csv`：提交文件
  - `over view.png`：结果截图

## 实验内容

1. 文本预处理：移除HTML标记、处理标点符号、数字和停用词
2. 词袋模型：使用CountVectorizer创建特征
3. 随机森林：训练分类模型
4. Word2Vec：训练分布式词向量模型
5. 生成提交文件：预测测试数据并生成符合Kaggle要求的提交文件

## 依赖

- pandas
- numpy
- scikit-learn
- beautifulsoup4
- nltk
- gensim
