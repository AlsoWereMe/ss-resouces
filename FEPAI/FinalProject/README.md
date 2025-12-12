# README

## Overview

本项目实现一个benchmark并用其编写期末论文。

任务定义：给定一条电影评论文本 $x$，模型需要判断其情感极性 $y \in \{positive, negative\}$。

评估对象：模型对文本的情感理解能力随长度的变化，也即模型对文本长度在情感能力上的鲁棒性。

数据集：IMDb大型电影评论数据集二次处理后的数据，以三分线与七分线区分为短、长两个文本数据集。

评估指标：$F1_{short},F1_{long},Drop=F1_{short}-F1_{long}$

模型分组：经典小模型、传统深度模型、预训练模型与LLM。

## TODO Roadmap

### 数据处理

- [x] 下载 IMDb 数据
- [x] 清洗文本
- [x] 计算每条 review 的长度（按 word 或 token）
- [x] 得到 short / long 两个数据集
- [x] 保存 processed 数据

### 模型

- [x] baseline: TF-IDF + LR/SVM/MNB
- [ ] Neural Models: BiLSTM, TextCNN
- [ ] Pretrained Transformers: RoBERTa-base, BERT-base
- [ ] LLMs: Qwen-7B,Meta Llama3 8B Instruct

### 评估

- [x] 实现统一 evaluation 函数（F1）
- [x] 分别对 short / long 数据进行评测
- [x] 计算 Drop
- [ ] 记录训练耗时、推理速度

### 可视化

- [ ] $F1_{short} \ vs. \ F1_{long} $柱状图
- [ ] $Drop$ 柱状图
- [ ] 错误案例展示

### 文档 & PPT

- [ ] 做 12 分钟 PPT
- [ ] 写论文大纲
- [ ] 写论文