# Project2 实验报告

## Transformer

### attention.py

#### ScaledDotProductAttention

该类需要实现缩放点积注意力机制，代码已经给出了K维度的获取，第二步计算Q与K的转置的点积

```python
scores = torch.matmul(query, key.transpose(-2,-1))
```

然后进行缩放

```python
scores = scores / math.sqrt(d_k)
```

再应用掩码

```python
if mask is not None:
	scores = scores.masked_fill(mask == 0, float('-1e9'))
```

用softmax算注意力权重

```python
attn_weights = torch.softmax(scores, dim=-1)
```

将dropout后的权重与V相乘即得输出

```python
output = torch.matmul(attn_weights, value)
```

#### MultiHeadAttention

该类需要实现多头注意力，在初始化QKV的投影层与输出层时，它们的维度与模型的总维度应当相同

```python
self.w_q = nn.Linear(d_model, d_model)
self.w_k = nn.Linear(d_model, d_model)
self.w_v = nn.Linear(d_model, d_model)
self.fc = nn.Linear(d_model, d_model)
```

再实现前向传播，将QKV投影到线性层的方法已经给出，将它们拆分为多个头

```python
query = query.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
key = key.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
value = value.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
```

调用刚刚实现的缩放点积注意力计算

```python
context = self.attention(query, key, value, mask=mask)
```

计算完成后合并多头为单头

```python
context = context.transpose(1, 2).contiguous().view(batch_size, len_q, self.d_model)
```

将输出进行全连接、dropout与添加残差，最后将其进行层归一化即可

```python
output = self.fc(context)
output = self.dropout(output)
output = self.layer_norm(output + residual)
```

### 	layers.py

#### PositionwiseFeedForward

该类需要实现前馈网络，初始化时定义线性层

```python
self.w_1 = nn.Linear(d_model, d_ff)
self.w_2 = nn.Linear(d_ff, d_model)
```

然后实现前向传播，将输出通过方才定义的第一层线性层并使用relu激活

```python
# 1. 通过第一个线性层，然后是ReLU激活函数。
output = self.relu(self.w_1(x))
```

最后通过第二个线性层并添加残差链接与层归一化

```python
# 2. 通过第二个线性层。
output = self.w_2(output)
output = self.dropout(output)
        
# 3. Add & Norm: 添加残差连接并应用 Layer Normalization。
output = self.layer_norm(output + residual)	
```

#### PositionalEncoding

该类需要实现位置编码，在初始化时需要计算位置编码矩阵pe，首先创建position张量

```python
position = torch.arange(0, max_len).unsqueeze(1)
```

然后计算除法项

```python
div_term = torch.exp(
    torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
)
```

最后区分奇偶索引

```python
# 3. 为偶数索引应用 sin 函数: pe[:, 0::2] = sin(position * div_term)
pe[:, 0::2] = torch.sin(position * div_term)
# 4. 为奇数索引应用 cos 函数: pe[:, 1::2] = cos(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

后续步骤代码已经给出，forward时只需要将初始化时计算的位置编码添加到输入上即可
```python
x = x + self.pe[:, :x.size(1), :]
```

#### LayerNorm

该类需要实现层归一化，在forward中简单利用公式计算即可

```python
# 1. 在最后一个维度 (d_model) 上计算均值和方差。
mean = x.mean(-1, keepdim=True)
std = x.var(-1, keepdim=True)
# 2. 归一化 x。
normalized_x = (x - mean) / torch.sqrt(std + self.eps)
# 3. 应用可学习的参数 gamma 和 beta。
output = self.gamma * normalized_x + self.beta
```

### blocks.py

#### EncoderBlock

该类为Transformer的编码块，每个编码块初始化时需要实例化多头自注意力层和位置前馈网络层

```python
# 实例化多头自注意力层和位置前馈网络层。
self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
```

前向传播时，因为前面已经实现多头注意力与前馈网络，简单调用实例化的组件即可

```python
# 1. 通过多头自注意力层。注意 Q, K, V 都来自 src。
src = self.self_attn(src, src, src, src_mask)
        
# 2. 通过位置前馈网络。
src = self.feed_forward(src)
```

#### DecoderBlock

该类为Transformer的解码块，每个解码块初始化时要实例化自注意力、交叉注意力与位置前馈网络

```python
# 实例化三个核心组件
self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
```

同样，前向传播时简单调用实例化的核心组件即可

```python
# 1. 掩码多头自注意力。Q, K, V 都来自 tgt，使用 tgt_mask。
tgt = self.self_attn(tgt, tgt, tgt, tgt_mask)
        
# 只有在 enc_src (Encoder的输出) 被提供时，才执行交叉注意力。
if enc_src is not None:
    # 2. 多头交叉注意力。Q 来自上一步的输出，K 和 V 来自 encoder 的输出 enc_src。
    #    使用 src_mask。
    tgt = self.cross_attn(tgt, enc_src, enc_src, src_mask)
        
# 3. 位置前馈网络。
tgt = self.feed_forward(tgt)
```

### model.py

该文件组织整个Transformer，分为三个类

#### TransformerEncoder

初始化时，堆叠多个编码块

```python
self.layers = nn.ModuleList([
    EncoderBlock(d_model, n_heads, d_ff, dropout)
    for _ in range(num_layers)
])
```

前向传播时将源序列传入即可

```python
# 依次将 src 通过 ModuleList 中的每一个 EncoderBlock。
for layer in self.layers:
    src = layer(src, src_mask)
return self.layer_norm(src)
```

#### TransformerDecoder

同理编码块，只将类改为DecoderBlock、参数src改为tgt即可

#### Transformer

初始化时实例化模型组件

```python
# 实例化模型的各个组件
self.src_embedding = nn.Embedding(src_vocab_size, d_model) # 源语言的词嵌入层
self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model) # 目标语言的词嵌入层
self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)   # 位置编码器
        
self.encoder = TransformerEncoder(num_layers, d_model, n_heads, d_ff, dropout)   # TransformerEncoder
self.decoder = TransformerDecoder(num_layers, d_model, n_heads, d_ff, dropout)   # TransformerDecoder
        
self.fc_out = nn.Linear(d_model, tgt_vocab_size)        # 最后的线性层，映射到目标词汇表大小
```

前向传播时进行Transformer流程，先进行嵌入与编码

```python
# 1. 嵌入和位置编码
src_processed = self.pos_encoder(
    self.src_embedding(src) * math.sqrt(self.d_model)
)
tgt_processed = self.pos_encoder(
    self.tgt_embedding(tgt) * math.sqrt(self.d_model)
)
```

然后是编码器与解码器

```python
# 2. Encoder
enc_output = self.encoder(src_processed, src_mask)
# 3. Decoder
dec_output = self.decoder(tgt_processed, enc_output, tgt_mask, src_mask)
```

最后将解码器输出送入线性层即可得到最后的结果

```python
# 4. 最终线性层
output = self.fc_out(dec_output)
```

#### @staticmethod

需要构造填充掩码

```python
mask = (seq != pad_idx)
return mask.unsqueeze(1).unsqueeze(2)
```

与构造因果掩码

```python
mask = torch.tril(torch.ones((size, size), device=device)).bool()
return mask.unsqueeze(0).unsqueeze(0)
```

### test

单元测试截图如下

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251117220335318.png" alt="image-20251117220335318" style="zoom:50%;" />

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251117220350110.png" alt="image-20251117220350110" style="zoom:50%;" />

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251117220933892.png" alt="image-20251117220933892" style="zoom:50%;" />

单元测试全部通过

## ViT

### vit.py

#### ViT

需要实现特征提取流程与前向传播，先获取预先定义的batch大小

```python
batch_size = x.size(0)
```

然后将图像转为块嵌入

```python
x = self.patch_embedding(x)
```

扩展CLS词元后将其拼接到块嵌入序列的开头

```python
cls_tokens = self.cls_token.expand(batch_size, -1, -1)
x = torch.cat([cls_tokens, x], dim=1)
```

对序列添加位置编码、应用dropout后输入编码器

```python
x = x + self.pos_embedding
```

前向传播时，需首先获取输出特征

```pyhton
features = self.forward_features(x, mask)
```

然后根据模型的用途选择返回值，如果是分类，提取输出

```python
cls_output = features[:, 0]
logits = self.mlp_head(cls_output)
return logits
```

而如果是特征提取，返回特征即可

```python
return features
```

### train_vit.py & predict_vit.py

train_vit.py与predict_vit.py按照提示完成即可

```python
# 1. 清空之前的梯度
optimizer.zero_grad()
            
# 2. 模型前向传播，获取输出
outputs = model(images)
            
# 3. 计算损失
loss = criterion(outputs, labels)
            
# 4. 反向传播，计算梯度
loss.backward()
            
# 5. 更新模型参数
optimizer.step()
```

```python
img_tensor = transform(img).unsqueeze(0).to(device)

logits = model(img_tensor)

predicted_class = data_cfg['class_names'][predicted_idx.item()]
```

### task

训练后，得到的结果为

![image-20251118205003889](C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251118205003889.png)

其训练时train&test Loss曲线图为

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251118210454717.png" alt="image-20251118210454717" style="zoom: 33%;" />

其accuracy曲线图为

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251118210427013.png" alt="image-20251118210427013" style="zoom: 33%;" />

使用如下的图片进行预测，

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251118215226091.png" alt="image-20251118215226091" style="zoom: 25%;" />

得到的结果为

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251118215140963.png" alt="image-20251118215140963" style="zoom: 50%;" />

## LM

### tokenizer.py

只需要将输入的字符串转为序列即可

```python
return [self.char_to_idx.get(c, self.unk_token_id) for c in text_string]
```

### llm.py

需要将嵌入向量前向传播，首先添加位置嵌入

```python
x = input_embeddings + pos_emb # 添加位置嵌入
x = self.drop_emb(x) # 应用 Dropout
```

然后创建因果掩码

```python
tgt_mask = torch.tril(torch.ones(T, T, device=input_embeddings.device))
tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, T, T)
```

最后依次通过所有Transformer块并应用归一化层

```python
x = self.final_norm(x)
logits = self.out_head(x)
```

在前向传播时转换为词元传入即可

```python
tok_emb = self.token_embedding(idx)  # (B, T, D)
logits = self.forward_from_embeddings(tok_emb)
```



### train_llm.py

这个类实现的过程和ViT基本一致

### generate_text.py

生成文本的脚本，按照提示完成即可

```python
# 1. 获取模型的 logits 输出
logits = model(context_cond)

# 2. 提取最后一个时间步的 logits。
logits_last_step = logits[:, -1, :]   # (B, C)

# 3. 将 logits 转换为概率分布。
probs = F.softmax(logits_last_step, dim=-1)

# 4. 从概率分布中采样一个词元。
next_token_idx = torch.multinomial(probs, num_samples

# 5. 将新生成的词元拼接到上下文中，为下一次迭代做准备
context = torch.cat([context, next_token_idx], dim=1)
```

### task

同样需要修改参数加快训练进度，训练过程中loss曲线图如下

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251119163712501.png" alt="image-20251119163712501" style="zoom:33%;" />

生成文本如下

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251119170307611.png" alt="image-20251119170307611" style="zoom: 67%;" />

## MM

### connector.py

构建多层感知机

```python
self.model = nn.Sequential(
    nn.Linear(vision_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, language_dim),
)
```

然后前向传播

```python
return self.model(x)
```

### mllm.py

需要实现前向传播与自回归生成

前向传播按照提示提取特征再投影拼接即可

```python
# 1. 从图像中提取视觉特征。
visual_features = self.vision_encoder.forward_features(images)
# 2. 使用 Connector 将视觉特征投影到语言模型的嵌入空间。
visual_embeddings = self.connector(visual_features)    
# 3. 获取文本的嵌入。
text_embeddings = self.language_model.token_embedding(text_tokens)
# 4. [关键步骤] 拼接视觉嵌入和文本嵌入。
inputs_embeddings = torch.cat([visual_embeddings, text_embeddings], dim=1)
# 5. 将拼接后的嵌入传入 LLM。
logits = self.language_model.forward_from_embeddings(inputs_embeddings)
        
```

自回归生成循环如下

```python
# a. 从当前的 `input_embeddings` 获取 logits。
logits = self.language_model.forward_from_embeddings(input_embeddings) 
            
# b. 只取序列中最后一个时间步的 logits，因为我们只关心预测下一个 token。
next_token_logits = logits[:, -1, :] 
            
# (Top-k 和 temperature 缩放已完成)
if top_k is not None:
    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
next_token_logits = next_token_logits / temperature
            
# c. 将 logits 转换为概率分布。
next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
# d. 从概率分布中采样一个 token。
next_token = torch.multinomial(next_token_probs, num_samples=1)
```

### train_mllm.py

训练组件，先为训练准备输入与目标

```python
model_input_text = captions[:, :-1]
targets = captions[:, 1:]
```

然后计算损失

```python
# Step 1: 创建 labels，全部填为 ignore_index
labels = torch.full(
    size=(logits.size(0), logits.size(1)),
    fill_value=criterion.ignore_index,
    device=logits.device
)

# Step 2: 文本标签放置位置（从视觉 token 长度开始）
label_start_idx = num_visual_tokens
label_end_idx = num_visual_tokens + targets.size(1)

# Step 3: 贴上 targets
labels[:, label_start_idx:label_end_idx] = targets

# Step 4: 计算 loss
loss = criterion(
    logits.view(-1, logits.size(-1)),
    labels.view(-1)
)

```

最后评估

```python
labels = torch.full(
    size=(logits.size(0), logits.size(1)),
    fill_value=criterion.ignore_index,
    device=logits.device
)

label_start_idx = num_visual_tokens
label_end_idx = num_visual_tokens + targets.size(1)

labels[:, label_start_idx:label_end_idx] = targets

loss = criterion(
    logits.view(-1, logits.size(-1)),
    labels.view(-1)
)
```

### inference_mllm.py

预测组件，先加载模型

```python
state_dict = torch.load(model_path, map_location=device)
mllm.load_state_dict(state_dict)
```

然后准备张量

```python
image_tensor = transform(image).unsqueeze(0).to(device)
```

最后生成描述

```python
generated_text = mllm.generate(
    image=image_tensor,
    prompt=prompt,
    max_new_tokens=infer_cfg["max_new_tokens"],
    temperature=infer_cfg["temperature"],
    top_k=infer_cfg.get("top_k", None)
)
```

### task

运行后效果图如下

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251119223207103.png" alt="image-20251119223207103" style="zoom:50%;" />

用两张图示范模型效果

图1

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251120003616418.png" alt="image-20251120003616418" style="zoom: 50%;" />

描述为

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251120003608731.png" alt="image-20251120003608731" style="zoom: 67%;" />

图2

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251120003851320.png" alt="image-20251120003851320" style="zoom:50%;" />

描述为

<img src="C:\Users\PATHF\AppData\Roaming\Typora\typora-user-images\image-20251120003930116.png" alt="image-20251120003930116" style="zoom: 67%;" />

## Issue & Fixed

有一些源代码的问题

- 如root_dir、patch_size这些参数，在运行时会报KeyError，代表没有在对应的config配置文件中提供，需要手动补上
- 类ViT在调用、构建时偶尔会缺少一些参数，运行时会报TypeError，需要在源码中填上去

还有一些训练效果问题

- 训练ViT需要的时间非常久并且Accuracy会停滞在10.00％，查阅资料发现是由于参数规模过大而数据集较小导致的效果不佳，遂降低参数规模后能成功训练，这个问题在LLM的训练上也有出现
