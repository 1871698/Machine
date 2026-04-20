# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：张晓敏
- **学号**：112304260116
- **班级**：数据1231


---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/c/word2vec-nlp-tutorial
- **提交日期**：2026-04-20

- **GitHub 仓库地址**：https://github.com/1871698/Machine
- **GitHub README 地址**：https://github.com/1871698/Machine/blob/main/README.md

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：0.95788
- **Private Score**（如有）：0.95788
- **排名**（如能看到可填写）：

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。
![alt text](112304260116_张晓敏_kaggle_score.png)

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。
> 截图文件名示例：`112304260116_张晓敏_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**
- 移除HTML标记：使用BeautifulSoup包解析HTML，提取纯文本内容
- 处理标点符号、数字和特殊符号：使用正则表达式将所有非字母字符替换为空格
- 移除停用词：使用NLTK的停用词列表，但保留否定词（如not、no等）以提高情感分析准确性
- 转小写：将所有文本转换为小写，统一格式
- 词形还原：尝试使用WordNetLemmatizer进行词形还原，失败时使用简单词干提取
- 清理特殊符号：多次清理反斜杠、引号、井号等特殊字符，确保数据彻底清洗

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**
- 自己训练Word2Vec模型：使用训练数据和未标记数据训练skip-gram模型
- 模型参数：
  - 架构：skip-gram (sg=1)
  - 训练算法：分层softmax (hs=1)
  - 高频词降采样：0.001 (sample=1e-3)
  - 词向量维度：400 (vector_size=400)
  - 上下文/窗口大小：15个词 (window=15)
  - 工作线程数：4 (workers=4)
  - 最小词数：40 (min_count=40)
- 句子向量表示：对每个评论中的所有单词向量取平均值，得到句子级别的embedding表示
- 词汇表大小：8175个单词

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**
- 尝试了多种分类模型：
  - 随机森林：准确率 0.8374
  - 逻辑回归 + 词袋：准确率 0.8616
  - 逻辑回归 + TF-IDF：准确率 0.8868
  - 逻辑回归 + Word2Vec (300d)：准确率 0.8662
  - 逻辑回归 + Word2Vec (400d, 15window)：准确率 0.8708
  - 逻辑回归 + Word2Vec + TF-IDF：准确率 0.8886
  - 线性回归 + Word2Vec + TF-IDF：准确率 0.8580

- 最终采用模型：**逻辑回归 + Word2Vec + TF-IDF**
  - 特征：400维Word2Vec均值embedding + 5000维TF-IDF特征
  - 准确率：0.8886
  - 优势：结合了Word2Vec的语义信息和TF-IDF的词频信息，性能最优

---

## 7. 实验流程
请简要说明你的实验流程。

示例：
1. 读取训练集和测试集
2. 对文本进行预处理
3. 训练或加载 Word2Vec 模型
4. 将每条文本表示为句向量
5. 用训练集训练分类器
6. 在测试集上预测结果
7. 生成 submission 文件并提交 Kaggle

**我的实验流程：**
1. 读取训练集（labeledTrainData.tsv）和测试集（testData.tsv）
2. 对文本进行预处理：
   - 移除HTML标记、标点符号、数字和特殊符号
   - 移除停用词（保留否定词）
   - 转小写、词形还原
   - 生成清洗后的数据文件（processedTrainData.tsv）
3. 训练Word2Vec模型：
   - 使用skip-gram架构，400维向量，15窗口大小
   - 在训练数据和未标记数据上训练
   - 保存模型供后续使用
4. 特征提取：
   - Word2Vec特征：对每个评论生成单词向量的平均值（400维）
   - TF-IDF特征：使用n-gram模式（1-2 gram），5000维特征
   - 结合两种特征形成5400维特征向量
5. 模型训练：
   - 使用逻辑回归分类器
   - 在训练集上训练，使用20%数据作为验证集
   - 模型准确率达到0.8886
6. 在测试集上预测结果：
   - 对测试数据进行相同的预处理和特征提取
   - 使用训练好的模型进行预测
7. 生成submission文件：
   - 生成符合Kaggle要求的提交文件（id, sentiment格式）
   - 确保ID列无引号，格式正确
8. 提交到Kaggle平台进行评分
9. 记录实验日志，将所有代码和结果上传到GitHub仓库

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

示例：
- `data/`：存放数据文件
- `src/`：存放源代码
- `notebooks/`：存放实验 notebook
- `images/`：存放 README 中使用的图片
- `submission/`：存放提交文件

**我的项目结构：**
```text
Machine/
├─ data_raw/                      # 原始数据文件
│  ├─ labeledTrainData.tsv        # 训练数据
│  ├─ testData.tsv                # 测试数据
│  └─ unlabeledTrainData.tsv      # 未标记数据
├─ src/                           # 源代码
│  ├─ make_submission.py          # 传统/集成方案（w2v/tfidf/nbsvm/stack）
│  ├─ make_submission_transformer.py # Transformer 微调生成提交
│  └─ blend_submissions.py        # 融合多个 submission
├─ artifacts/                     # 传统模型缓存
├─ images/                        # README 截图
├─ code/                          # 原有实验代码
│  ├─ text_preprocessing.py       # 文本预处理
│  ├─ bag_of_words_forest.py      # 词袋模型和随机森林
│  ├─ train_word2vec_tuned.py     # 调整参数的Word2Vec模型训练
│  └─ ...                         # 其他实验代码
├─ results/                       # 实验结果
│  ├─ experiment_log.csv          # 实验日志
│  └─ sampleSubmission.csv        # 提交文件
├─ requirements.txt               # 依赖
├─ .gitignore                     # git忽略文件
└─ README.md                      # 仓库说明文档
```

---

## 9. 实验日志

本实验共进行了37次实验，主要改动和结果如下：

| ID | 主要改动 | 实现结果 |
|----|----------|----------|
| 1 | 初始文本预处理，移除HTML标记、标点符号、数字和停用词 | 生成processedTrainData.tsv文件 |
| 2-26 | 多次数据清洗，移除残留的特殊符号和引号 | 彻底清洗数据，确保无残留符号 |
| 27 | 使用词袋模型+随机森林训练模型 | 准确率0.8374 |
| 28 | 训练Word2Vec模型（skip-gram，300维，10窗口） | 词汇表大小8157 |
| 29 | 使用Word2Vec均值embedding+逻辑回归 | 准确率0.8662 |
| 30 | 调整Word2Vec参数（400维，15窗口） | 准确率0.8708 |
| 31 | 结合Word2Vec和TF-IDF特征 | 准确率0.8886 |
| 32 | 增强文本预处理功能，清理所有列 | 彻底清洗所有列的特殊符号和引号 |
| 33 | 基于清洗数据重新训练Word2Vec模型 | 词汇表大小8175 |
| 34 | 基于清洗数据测试Word2Vec+逻辑回归 | 准确率0.8708 |
| 35 | 基于清洗数据测试Word2Vec+TF-IDF+逻辑回归 | 准确率0.8886 |
| 36 | 将逻辑回归改为线性回归 | 准确率0.8580 |
| 37 | 增强文本预处理功能，添加词形还原 | 使用简单词干提取，提高文本处理质量 |
| 38 | 按照GitHub仓库方法重新组织项目结构 | 创建data_raw、src、artifacts等目录 |
| 39 | 添加Transformer模型支持 | 实现make_submission_transformer.py |
| 40 | 运行传统强基线（TF-IDF）模型 | 生成submission_tfidf_both.csv，准确率0.8950 |

---

## 10. 复现方式

### 10.1 安装依赖
```bash
cd D:\机器学习（大三下）\实验2\labeledTrainData.tsv
pip install -r .\requirements.txt
```

### 10.2 传统强基线（TF-IDF）
```bash
python .\src\make_submission.py --data-dir .\data_raw --features tfidf_both --out .\submission_tfidf_both.csv
```

### 10.3 Word2Vec模型
```bash
python .\src\make_submission.py --data-dir .\data_raw --features word2vec --out .\submission_word2vec.csv
```

### 10.4 Transformer模型（需GPU）
```bash
python .\src\make_submission_transformer.py --data-dir .\data_raw --model roberta-large --epochs 2 --batch-size 2 --grad-accum 8 --fp16 --grad-checkpointing --valid-ratio 0 --no-eval --out .\submission_roberta_large.csv
```

### 10.5 融合提交
```bash
python .\src\blend_submissions.py --method rank_mean --out .\submission_blend.csv --inputs .\submission_tfidf_both.csv .\submission_word2vec.csv
```

---

## 10. 模型性能对比

| 方法 | 准确率 | 特征类型 |
|------|--------|----------|
| 随机森林 + 词袋 | 0.8374 | 词频向量 |
| 逻辑回归 + 词袋 | 0.8616 | 词频向量 |
| 逻辑回归 + TF-IDF | 0.8868 | TF-IDF向量 |
| 逻辑回归 + Word2Vec (300d) | 0.8662 | 300维词向量均值 |
| 逻辑回归 + Word2Vec (400d, 15window) | 0.8708 | 400维词向量均值 |
| 逻辑回归 + Word2Vec + TF-IDF | **0.8886** | 400维词向量均值 + 5000维TF-IDF |
| 线性回归 + Word2Vec + TF-IDF | 0.8580 | 400维词向量均值 + 5000维TF-IDF |

**最佳模型**：逻辑回归 + Word2Vec + TF-IDF，准确率0.8886

---

## 11. 总结与展望

### 总结
本实验成功实现了基于Word2Vec的情感分析系统，通过多次实验和优化，最终达到了0.8886的准确率。主要成果包括：
- 完善了文本预处理流程，彻底清洗数据
- 训练了高质量的Word2Vec模型
- 结合多种特征表示方法，提高了模型性能
- 尝试了多种分类模型，找到了最优方案

### 展望
为进一步提高模型性能，可以考虑以下方向：
1. 使用预训练词向量（如GloVe、FastText）
2. 尝试深度学习模型（如LSTM、CNN、Transformer）
3. 使用集成学习方法（如XGBoost、LightGBM）
4. 进行超参数调优（网格搜索、随机搜索）
5. 数据增强（同义词替换、回译等）
6. 使用更强大的预训练语言模型（如BERT、RoBERTa）

---

**实验完成日期**：2026-04-17
**学生签名**：张晓敏
**学号**：112304260116